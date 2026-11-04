use collections::{HashMap, HashSet};
use gpui::{
    AnyElement, App, AppContext, Context, Hsla, ImageSource, IntoElement, Pixels, RenderImage,
    SMOOTH_SVG_SCALE_FACTOR, SharedString, Styled, StyledText, Task, img, px, size,
};
use theme::ActiveTheme;

use ratex_layout::{LayoutOptions, layout, to_display_list, validate_layout};
use ratex_parser::parser::parse;
use ratex_svg::{SvgColorSyntax, SvgOptions, render_to_svg_with_color_syntax};
use ratex_types::color::Color;
use ratex_types::math_style::MathStyle;

use std::sync::{Arc, OnceLock};

use crate::parser::MarkdownEvent;
use crate::{Markdown, ParsedMarkdown};

/// The size of one `em` of math, in the user units of the generated SVG.
///
/// `ratex-svg` emits dimensions in `pt`, which `resvg` (used by gpui's SVG
/// renderer) resolves at 96 DPI. This value is chosen so that one `em` of math
/// comes out close to the size of the surrounding body text once rasterized.
const MATH_FONT_SIZE: f64 = 10.5;
/// Padding around a formula, in the same units as [`MATH_FONT_SIZE`].
const MATH_PADDING: f64 = 1.0;

/// A single math expression, keyed in the render cache by its source and
/// whether it is displayed as a block or inline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MathExpression {
    pub(crate) latex: SharedString,
    pub(crate) display: bool,
}

impl MathExpression {
    pub(crate) fn new(latex: SharedString, display: bool) -> Self {
        Self { latex, display }
    }
}

#[derive(Clone, Default)]
pub(crate) struct MathState {
    cache: HashMap<MathExpression, Arc<CachedMath>>,
    /// The text color the cached rasters were rendered with. When the theme
    /// changes, the cache is dropped so formulas are re-rendered in the new
    /// color.
    color: Option<Hsla>,
}

struct CachedMath {
    render_image: Arc<OnceLock<anyhow::Result<Arc<RenderImage>>>>,
    _task: Task<()>,
}

impl MathState {
    /// Reconciles the cache against the math expressions found in `parsed`,
    /// spawning background renders for any expression that is not cached yet.
    pub(crate) fn update(&mut self, parsed: &ParsedMarkdown, cx: &mut Context<Markdown>) {
        let color = cx.theme().colors().text;
        if self.color != Some(color) {
            self.color = Some(color);
            self.clear(cx);
        }

        let mut expressions = Vec::new();
        for (_, event) in parsed.events.iter() {
            match event {
                MarkdownEvent::InlineMath(latex) => {
                    expressions.push(MathExpression::new(latex.clone(), false));
                }
                MarkdownEvent::DisplayMath(latex) => {
                    expressions.push(MathExpression::new(latex.clone(), true));
                }
                _ => {}
            }
        }

        for expression in &expressions {
            if !self.cache.contains_key(expression) {
                self.cache.insert(
                    expression.clone(),
                    Arc::new(CachedMath::new(expression.clone(), color, cx)),
                );
            }
        }

        let wanted: HashSet<MathExpression> = expressions.into_iter().collect();
        self.cache.retain(|expression, cached| {
            let keep = wanted.contains(expression);
            if !keep {
                cached.drop_image(cx);
            }
            keep
        });
    }

    pub(crate) fn clear(&mut self, cx: &mut App) {
        for cached in self.cache.values() {
            cached.drop_image(cx);
        }
        self.cache.clear();
    }

    fn get(&self, expression: &MathExpression) -> Option<&Arc<CachedMath>> {
        self.cache.get(expression)
    }
}

impl CachedMath {
    fn new(expression: MathExpression, color: Hsla, cx: &mut Context<Markdown>) -> Self {
        let render_image = Arc::new(OnceLock::<anyhow::Result<Arc<RenderImage>>>::new());
        let svg_renderer = cx.svg_renderer();

        let task = cx.spawn({
            let render_image = render_image.clone();
            async move |this, cx| {
                let rendered = cx
                    .background_spawn(async move {
                        let svg = render_latex_to_svg(&expression.latex, expression.display, color)?;
                        let tree = svg_renderer
                            .parse_svg(svg.as_bytes())
                            .map_err(|error| anyhow::anyhow!("{error}"))?;
                        let image = svg_renderer
                            .render_parsed(&tree, SMOOTH_SVG_SCALE_FACTOR)
                            .map_err(|error| anyhow::anyhow!("{error}"))?;
                        anyhow::Ok(image)
                    })
                    .await;
                render_image.set(rendered).ok();
                this.update(cx, |_, cx| cx.notify()).ok();
            }
        });

        Self {
            render_image,
            _task: task,
        }
    }

    fn drop_image(&self, cx: &mut App) {
        if let Some(Ok(render_image)) = self.render_image.get() {
            cx.drop_image(render_image.clone(), None);
        }
    }
}

/// Renders a math expression to a self-contained SVG document.
///
/// The `embed-fonts` feature of `ratex-svg` bakes the KaTeX glyph outlines into
/// the SVG as paths, so the result does not depend on any fonts being installed
/// on the host. The text color is baked in as well, since the raster produced
/// from the SVG cannot be re-tinted.
fn render_latex_to_svg(latex: &str, display: bool, color: Hsla) -> anyhow::Result<String> {
    let ast = parse(latex).map_err(|error| anyhow::anyhow!("failed to parse math: {error}"))?;
    let style = if display {
        MathStyle::Display
    } else {
        MathStyle::Text
    };
    let rgba = color.to_rgb();
    let math_color = Color::new(rgba.r, rgba.g, rgba.b, rgba.a);
    let layout_options = LayoutOptions::default()
        .with_style(style)
        .with_color(math_color);
    let layout_box = layout(&ast, &layout_options);
    validate_layout(&layout_box)
        .map_err(|error| anyhow::anyhow!("failed to lay out math: {error}"))?;
    let display_list = to_display_list(&layout_box);

    // In debug builds `ratex-katex-fonts` (rust-embed) loads font files from the
    // cargo registry crate directory at runtime, so a debug binary built elsewhere
    // silently falls back to `<text>` and renders wrong glyphs; release embeds them.
    let svg_options = SvgOptions {
        font_size: MATH_FONT_SIZE,
        padding: MATH_PADDING,
        stroke_width: 1.5,
        embed_glyphs: true,
        font_dir: String::new(),
    };
    let svg = render_to_svg_with_color_syntax(
        &display_list,
        &svg_options,
        SvgColorSyntax::Rgba,
    );

    Ok(svg)
}

/// Renders a cached math expression as an image, falling back to the raw source
/// text while the background render is pending or if it failed.
///
/// The formula is scaled so that one em of math matches `body_font_size`, the
/// size of the surrounding text, instead of the fixed size it was rasterized at.
pub(crate) fn render_math(
    state: &MathState,
    expression: &MathExpression,
    source_text: SharedString,
    body_font_size: Pixels,
) -> AnyElement {
    match state
        .get(expression)
        .and_then(|cached| cached.render_image.get())
    {
        Some(Ok(render_image)) => {
            let scale = body_font_size / math_em_size();
            let base_size = math_base_size(render_image);
            let base_size = size(base_size.width * scale, base_size.height * scale);
            img(ImageSource::Render(render_image.clone()))
                .w(base_size.width)
                .h(base_size.height)
                .into_any_element()
        }
        _ => StyledText::new(source_text).into_any_element(),
    }
}

/// The on-screen size of one em of math at the rasterized [`MATH_FONT_SIZE`].
/// `ratex-svg` declares its dimensions in `pt`, which are 4/3 of a pixel.
fn math_em_size() -> Pixels {
    px((MATH_FONT_SIZE * 4.0 / 3.0) as f32)
}

/// Recovers the formula's logical size from its raster.
///
/// Rasters from `SvgRenderer::render_parsed` are produced at
/// [`SMOOTH_SVG_SCALE_FACTOR`] times the SVG's declared size, so dividing the
/// device size by it recovers the declared size regardless of the display
/// scale.
fn math_base_size(image: &RenderImage) -> gpui::Size<Pixels> {
    let device_size = image.size(0);
    size(
        px(device_size.width.0 as f32 / SMOOTH_SVG_SCALE_FACTOR),
        px(device_size.height.0 as f32 / SMOOTH_SVG_SCALE_FACTOR),
    )
}

#[cfg(test)]
mod tests {
    use super::render_latex_to_svg;
    use gpui::hsla;

    #[test]
    fn test_render_latex_to_svg_produces_path_glyphs() {
        let svg = render_latex_to_svg(r"\frac{a}{b}", true, hsla(0.0, 0.0, 0.0, 1.0))
            .expect("fraction should render");
        assert!(svg.starts_with("<svg"), "expected an SVG document");
        assert!(svg.contains("<path"), "expected glyphs to be outlined");
        assert!(
            !svg.contains("<text"),
            "glyphs should be embedded, not left as text"
        );
    }

    #[test]
    fn test_render_latex_embeds_all_glyphs_as_paths() {
        let expressions = [
            r"\mathrm{softmax}(Q K^T)",
            r"\mathbb{R}",
            r"\frac{Q K^T}{\sqrt{d_k}}",
            r"\sum_{\ell=1}^{d_k} q_{i\ell} k_{j\ell}",
            r"Z_i \leftarrow Z_i + e_{ij} v_j",
            r"\left( \frac{a}{b} \right)",
            r"x \in \mathbb{R}^{3 \times 2}",
        ];
        for latex in expressions {
            let svg = render_latex_to_svg(latex, true, hsla(0.0, 0.0, 0.0, 1.0))
                .unwrap_or_else(|error| panic!("{latex}: {error}"));
            assert!(svg.contains("<path"), "{latex}: expected outlined glyphs");
            assert!(
                !svg.contains("<text"),
                "{latex}: glyphs must be embedded as paths"
            );
        }
    }
}
