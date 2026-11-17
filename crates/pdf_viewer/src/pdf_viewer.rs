use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use gpui::RenderImage;
use image::{Frame, RgbaImage};
use pdfium_render::prelude::*;
use smallvec::smallvec;

/// Loads the pdfium shared library from `directory`, which must contain the
/// platform library file (for example `libpdfium.so`).
pub fn bind_pdfium(directory: &Path) -> Result<Pdfium> {
    let library_path = Pdfium::pdfium_platform_library_name_at_path(directory);
    let bindings = Pdfium::bind_to_library(library_path)
        .context("failed to load the pdfium library")?;
    Ok(Pdfium::new(bindings))
}

/// Renders one page of a PDF document to an image `target_width` pixels wide,
/// keeping the page's aspect ratio.
pub fn render_page(
    pdfium: &Pdfium,
    document_bytes: &[u8],
    page_index: i32,
    target_width: i32,
) -> Result<Arc<RenderImage>> {
    let document = pdfium
        .load_pdf_from_byte_slice(document_bytes, None)
        .context("failed to open the PDF document")?;
    let page = document
        .pages()
        .get(page_index)
        .context("the PDF has no page at this index")?;
    let configuration = PdfRenderConfig::new().set_target_width(target_width);
    let rendered = page
        .render_with_config(&configuration)
        .context("failed to render the PDF page")?
        .as_image()
        .context("failed to convert the rendered page to pixels")?
        .into_rgba8();

    let width = rendered.width();
    let height = rendered.height();
    // gpui expects BGRA pixels, so the red and blue channels are swapped.
    let mut pixels = rendered.into_raw();
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    let buffer = RgbaImage::from_raw(width, height, pixels)
        .context("rendered page has an unexpected pixel buffer size")?;
    Ok(Arc::new(RenderImage::new(smallvec![Frame::new(buffer)])))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs `PDFIUM_LIBRARY_DIRECTORY` and `PDF_VIEWER_TEST_PDF` to point at a
    /// pdfium library directory and any PDF file; run with `--ignored`.
    #[test]
    #[ignore = "needs PDFIUM_LIBRARY_DIRECTORY and PDF_VIEWER_TEST_PDF"]
    fn test_render_first_page_of_pdf() {
        let library_directory = std::env::var("PDFIUM_LIBRARY_DIRECTORY")
            .expect("PDFIUM_LIBRARY_DIRECTORY must be set");
        let pdf_path = std::env::var("PDF_VIEWER_TEST_PDF").expect("PDF_VIEWER_TEST_PDF must be set");
        let pdfium = bind_pdfium(Path::new(&library_directory)).expect("bind pdfium");
        let bytes = std::fs::read(&pdf_path).expect("read test PDF");

        let image = render_page(&pdfium, &bytes, 0, 600).expect("render first page");
        let size = image.size(0);
        assert_eq!(size.width.0, 600);
        assert!(size.height.0 > 0);
    }
}
