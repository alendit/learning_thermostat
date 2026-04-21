from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "assets" / "brand"
OUT_DIR = ROOT / "custom_components" / "learning_thermostat" / "brand"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Futura.ttc"
FONT_BODY = "/System/Library/Fonts/Supplemental/GillSans.ttc"


@dataclass(frozen=True)
class Theme:
    name: str
    shell: str
    spine: str
    page: str
    accent: str
    text_primary: str
    text_secondary: str
    center_line: str


LIGHT = Theme(
    name="light",
    shell="#14354A",
    spine="#2F607A",
    page="#F5F1E9",
    accent="#E36B54",
    text_primary="#14354A",
    text_secondary="#4D7B79",
    center_line="#7DBFB3",
)

DARK = Theme(
    name="dark",
    shell="#F4F0E8",
    spine="#C9BA99",
    page="#17384D",
    accent="#FF8E72",
    text_primary="#F4F0E8",
    text_secondary="#8AC8BC",
    center_line="#8AC8BC",
)


def ensure_dirs() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def book_svg(theme: Theme) -> str:
    return f"""
      <g>
        <rect x="24" y="20" width="112" height="120" rx="24" fill="{theme.shell}"/>
        <rect x="38" y="20" width="14" height="120" rx="7" fill="{theme.spine}"/>
        <path d="M56 43c8-4 17-6 28-6 14 0 26 4 33 13 7-9 19-13 33-13v60c-14 0-26 4-33 12-7-8-19-12-33-12-11 0-20 2-28 6z" fill="{theme.page}"/>
        <rect x="75" y="53" width="10" height="30" rx="5" fill="{theme.shell}"/>
        <circle cx="80" cy="94" r="15" fill="{theme.shell}"/>
        <rect x="77" y="58" width="6" height="23" rx="3" fill="{theme.accent}"/>
        <circle cx="80" cy="94" r="7" fill="{theme.accent}"/>
        <path d="M80 48v57" stroke="{theme.center_line}" stroke-width="2.5" opacity="0.55"/>
      </g>
    """.strip()


def icon_svg(theme: Theme, size: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 160 160">{book_svg(theme)}</svg>\n'
    )


def logo_svg(theme: Theme, width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 512 160">
  <rect width="512" height="160" fill="none"/>
  <g transform="translate(8 0)">
    <g transform="translate(0 6) scale(0.95)">
      {book_svg(theme)}
    </g>
    <text x="146" y="72" font-family="Futura, Arial, sans-serif" font-size="40" font-weight="700" fill="{theme.text_primary}">Learning</text>
    <text x="146" y="112" font-family="Gill Sans, Trebuchet MS, sans-serif" font-size="34" font-weight="600" fill="{theme.text_secondary}">Thermostat</text>
  </g>
</svg>
"""


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: float,
    fill: str,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def draw_mark(
    img: Image.Image, theme: Theme, offset: tuple[float, float], scale: float
) -> None:
    draw = ImageDraw.Draw(img)
    ox, oy = offset

    def sx(value: float) -> float:
        return ox + value * scale

    def sy(value: float) -> float:
        return oy + value * scale

    rounded_rect(draw, (sx(24), sy(20), sx(136), sy(140)), 24 * scale, theme.shell)
    rounded_rect(draw, (sx(38), sy(20), sx(52), sy(140)), 7 * scale, theme.spine)

    left_page = [
        (sx(56), sy(43)),
        (sx(67), sy(38)),
        (sx(84), sy(39)),
        (sx(88), sy(54)),
        (sx(88), sy(103)),
        (sx(79), sy(98)),
        (sx(67), sy(96)),
        (sx(56), sy(102)),
    ]
    right_page = [
        (sx(84), sy(39)),
        (sx(97), sy(37)),
        (sx(112), sy(41)),
        (sx(121), sy(50)),
        (sx(121), sy(97)),
        (sx(111), sy(100)),
        (sx(97), sy(104)),
        (sx(88), sy(103)),
        (sx(88), sy(54)),
    ]
    draw.polygon(left_page, fill=theme.page)
    draw.polygon(right_page, fill=theme.page)
    draw.line(
        (sx(80), sy(48), sx(80), sy(105)),
        fill=theme.center_line,
        width=max(1, int(2.5 * scale)),
    )
    rounded_rect(draw, (sx(75), sy(53), sx(85), sy(83)), 5 * scale, theme.shell)
    draw.ellipse((sx(65), sy(79), sx(95), sy(109)), fill=theme.shell)
    rounded_rect(draw, (sx(77), sy(58), sx(83), sy(81)), 3 * scale, theme.accent)
    draw.ellipse((sx(73), sy(87), sx(87), sy(101)), fill=theme.accent)


def render_icon(theme: Theme, size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_mark(img, theme, (0, 0), size / 160)
    return img


def render_logo(theme: Theme, width: int, height: int) -> Image.Image:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_mark(img, theme, (0, 5), 0.95)
    draw = ImageDraw.Draw(img)
    learning_font = load_font(FONT_BOLD, 44)
    thermostat_font = load_font(FONT_BODY, 38)
    draw.text((146, 34), "Learning", font=learning_font, fill=theme.text_primary)
    draw.text((146, 78), "Thermostat", font=thermostat_font, fill=theme.text_secondary)
    return img


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    write(SRC_DIR / "icon-light.svg", icon_svg(LIGHT, 256))
    write(SRC_DIR / "icon-dark.svg", icon_svg(DARK, 256))
    write(SRC_DIR / "logo-light.svg", logo_svg(LIGHT, 512, 160))
    write(SRC_DIR / "logo-dark.svg", logo_svg(DARK, 512, 160))

    render_icon(LIGHT, 256).save(OUT_DIR / "icon.png")
    render_icon(DARK, 256).save(OUT_DIR / "dark_icon.png")
    render_logo(LIGHT, 512, 160).save(OUT_DIR / "logo.png")
    render_logo(DARK, 512, 160).save(OUT_DIR / "dark_logo.png")

    render_icon(LIGHT, 512).save(OUT_DIR / "icon@2x.png")
    render_icon(DARK, 512).save(OUT_DIR / "dark_icon@2x.png")
    render_logo(LIGHT, 1024, 320).save(OUT_DIR / "logo@2x.png")
    render_logo(DARK, 1024, 320).save(OUT_DIR / "dark_logo@2x.png")


if __name__ == "__main__":
    main()
