use collections::{HashMap, HashSet};
use gpui::{
    AnyElement, AppContext, Context, Hsla, IntoElement, Pixels, SharedString, Styled, StyledText,
    Task, TextStyle, Window, px, svg,
};

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
/// Dimensions are expressed in these units so that a formula scales with the
/// surrounding body text: one `em` of math is drawn at the body font size.
const MATH_FONT_SIZE: f64 = 10.5;
/// Padding around a formula, in the same units as [`MATH_FONT_SIZE`].
const MATH_PADDING: f64 = 1.0;
/// Scales formulas relative to the body text they sit in.
const MATH_SIZE_SCALE: f32 = 1.1;
/// Calibrated against a screenshot: inline formulas sat about 4px below the text
/// baseline at a body size of roughly 12px, which the metric-based offset alone
/// does not account for.
const INLINE_BASELINE_CORRECTION_EM: f32 = 0.33;

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
}

struct CachedMath {
    render: Arc<OnceLock<anyhow::Result<RenderedMath>>>,
    _task: Task<()>,
}

struct RenderedMath {
    svg: String,
    /// The formula's box, including padding, in [`MATH_FONT_SIZE`] units.
    width: f64,
    height: f64,
    /// The distance from the top of the box to the formula's baseline, in
    /// [`MATH_FONT_SIZE`] units.
    ascent: f64,
}

impl MathState {
    /// Reconciles the cache against the math expressions found in `parsed`,
    /// spawning background layouts for any expression that is not cached yet.
    pub(crate) fn update(&mut self, parsed: &ParsedMarkdown, cx: &mut Context<Markdown>) {
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
                    Arc::new(CachedMath::new(expression.clone(), cx)),
                );
            }
        }

        let wanted: HashSet<MathExpression> = expressions.into_iter().collect();
        self.cache
            .retain(|expression, _| wanted.contains(expression));
    }

    pub(crate) fn clear(&mut self) {
        self.cache.clear();
    }

    fn get(&self, expression: &MathExpression) -> Option<&Arc<CachedMath>> {
        self.cache.get(expression)
    }
}

impl CachedMath {
    fn new(expression: MathExpression, cx: &mut Context<Markdown>) -> Self {
        let render = Arc::new(OnceLock::<anyhow::Result<RenderedMath>>::new());

        let task = cx.spawn({
            let render = render.clone();
            async move |this, cx| {
                let rendered = cx
                    .background_spawn(async move {
                        render_latex_to_svg(&expression.latex, expression.display)
                    })
                    .await;
                render.set(rendered).ok();
                this.update(cx, |_, cx| cx.notify()).ok();
            }
        });

        Self {
            render,
            _task: task,
        }
    }
}

/// Lays out a math expression and renders it to a self-contained SVG document.
///
/// The `embed-fonts` feature of `ratex-svg` bakes the KaTeX glyph outlines into
/// the SVG as paths, so the result does not depend on any fonts being installed
/// on the host. The SVG is drawn with the surrounding text color, so no color is
/// baked into it.
fn render_latex_to_svg(latex: &str, display: bool) -> anyhow::Result<RenderedMath> {
    let ast = parse(latex).map_err(|error| anyhow::anyhow!("failed to parse math: {error}"))?;
    let style = if display {
        MathStyle::Display
    } else {
        MathStyle::Text
    };
    let layout_options = LayoutOptions::default()
        .with_style(style)
        .with_color(Color::new(0.0, 0.0, 0.0, 1.0));
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
    let svg = render_to_svg_with_color_syntax(&display_list, &svg_options, SvgColorSyntax::Rgba);

    Ok(RenderedMath {
        svg,
        width: display_list.width * MATH_FONT_SIZE + 2.0 * MATH_PADDING,
        height: (display_list.height + display_list.depth) * MATH_FONT_SIZE + 2.0 * MATH_PADDING,
        ascent: display_list.height * MATH_FONT_SIZE + MATH_PADDING,
    })
}

/// The distance from the top of a text row to its baseline, for the text style
/// the row is laid out with.
pub(crate) fn text_baseline_offset(text_style: &TextStyle, window: &Window) -> Pixels {
    let rem_size = window.rem_size();
    let font_size = text_style.font_size.to_pixels(rem_size);
    let line_height = text_style.line_height_in_pixels(rem_size);
    let font_id = window.text_system().resolve_font(&text_style.font());
    window
        .text_system()
        .baseline_offset(font_id, font_size, line_height)
}

/// Renders a cached math expression, falling back to the raw source text while
/// the background layout is pending or if it failed.
///
/// The formula is painted as an SVG at the size it occupies on screen, so the
/// glyphs are rasterized at the final resolution rather than scaled from a
/// fixed-size bitmap. It takes the surrounding text color.
pub(crate) fn render_math(
    state: &MathState,
    expression: &MathExpression,
    source_text: SharedString,
    body_font_size: Pixels,
    text_color: Hsla,
    text_baseline: Pixels,
) -> AnyElement {
    match state.get(expression).and_then(|cached| cached.render.get()) {
        Some(Ok(rendered)) => {
            let pixels_per_unit =
                f32::from(body_font_size) / MATH_FONT_SIZE as f32 * MATH_SIZE_SCALE;
            let formula = svg()
                .data(rendered.svg.as_bytes())
                .text_color(text_color)
                .w(px(rendered.width as f32 * pixels_per_unit))
                .h(px(rendered.height as f32 * pixels_per_unit));
            if expression.display {
                formula.into_any_element()
            } else {
                // Inline images are top-aligned in the text row, so the formula is moved
                // until its baseline lands on the text baseline.
                let formula_ascent = px(rendered.ascent as f32 * pixels_per_unit);
                let correction = body_font_size * INLINE_BASELINE_CORRECTION_EM;
                formula
                    .relative()
                    .top(text_baseline - formula_ascent - correction)
                    .into_any_element()
            }
        }
        _ => StyledText::new(source_text).into_any_element(),
    }
}

#[cfg(test)]
mod tests {
    use super::render_latex_to_svg;

    #[test]
    fn test_render_latex_to_svg_produces_path_glyphs() {
        let svg = render_latex_to_svg(r"\frac{a}{b}", true)
            .expect("fraction should render")
            .svg;
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
            let svg = render_latex_to_svg(latex, true)
                .unwrap_or_else(|error| panic!("{latex}: {error}"))
                .svg;
            assert!(svg.contains("<path"), "{latex}: expected outlined glyphs");
            assert!(
                !svg.contains("<text"),
                "{latex}: glyphs must be embedded as paths"
            );
        }
    }
}
