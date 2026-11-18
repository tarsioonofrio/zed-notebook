use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result, bail};

/// A position on a PDF page. `x` and `y` are PDF points measured from the top left
/// of the page, which is the convention `synctex` uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PdfPosition {
    pub page: u32,
    pub x: f32,
    pub y: f32,
    /// Width of the source line's box, or zero when the position came from a click.
    pub width: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourcePosition {
    pub path: PathBuf,
    pub line: u32,
}

/// Finds where a source line lands in the PDF. `line` is 1-based.
pub fn forward_search(source: &Path, line: u32, column: u32, pdf: &Path) -> Result<PdfPosition> {
    let input = format!("{line}:{column}:{}", source.display());
    let output = run_synctex(&["view", "-i", &input, "-o", &pdf.to_string_lossy()])?;
    let page = field(&output, "Page").context("synctex did not report a page")?;
    let x = field(&output, "x").context("synctex did not report an x position")?;
    let y = field(&output, "y").context("synctex did not report a y position")?;
    let width = field(&output, "W").context("synctex did not report a width")?;
    Ok(PdfPosition {
        page: page.parse().context("invalid synctex page")?,
        x: x.parse().context("invalid synctex x position")?,
        y: y.parse().context("invalid synctex y position")?,
        width: width.parse().context("invalid synctex width")?,
    })
}

/// Finds the source line behind a point on a PDF page. `page` is 1-based.
pub fn inverse_search(pdf: &Path, position: PdfPosition) -> Result<SourcePosition> {
    let location = format!(
        "{}:{}:{}:{}",
        position.page,
        position.x,
        position.y,
        pdf.display()
    );
    let output = run_synctex(&["edit", "-o", &location])?;
    let path = field(&output, "Input").context("synctex did not report a source file")?;
    let line = field(&output, "Line").context("synctex did not report a source line")?;
    Ok(SourcePosition {
        path: normalize_path(Path::new(path)),
        line: line.parse().context("invalid synctex line")?,
    })
}

fn run_synctex(arguments: &[&str]) -> Result<String> {
    let output = Command::new("synctex")
        .args(arguments)
        .output()
        .context("failed to run synctex; is it installed?")?;
    if !output.status.success() {
        bail!(
            "synctex failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn field<'a>(output: &'a str, name: &str) -> Option<&'a str> {
    output.lines().find_map(|line| {
        line.strip_prefix(name)
            .and_then(|rest| rest.strip_prefix(':'))
            .map(str::trim)
    })
}

/// `synctex` prints paths such as `dir/./file.tex`, which should compare equal to
/// the path the editor uses.
fn normalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
