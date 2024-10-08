#!/usr/bin/env python3
from pathlib import Path
import shutil
import textwrap

from jinja2 import Environment
from jinja2 import FileSystemLoader
from markupsafe import Markup
import pygments.formatters.html
import pygments.lexers.python

import pdoc.render

here = Path(__file__).parent

if __name__ == "__main__":
    demo1 = here / "demo1.py"
    demo2 = here / "demo2.py"
    demo3 = here / "demo3.py"
    env = Environment(
        loader=FileSystemLoader([here]),
        autoescape=True,
    )

    lexer = pygments.lexers.python.PythonLexer()
    formatter = pygments.formatters.html.HtmlFormatter(style="dracula")
    pygments_css = formatter.get_style_defs()
    example_html1 = Markup(
        pygments.highlight(demo1.read_text("utf8"), lexer, formatter)
    )
    example_html2 = Markup(
        pygments.highlight(demo2.read_text("utf8"), lexer, formatter)
    )
    example_html3 = Markup(
        pygments.highlight(demo3.read_text("utf8"), lexer, formatter).replace(
            "converged", '<span class="highlighted">converged</span>'
        )
    )

    (here / "index.html").write_bytes(
        env.get_template("index.html.jinja2")
        .render(
            example_html1=example_html1,
            example_html2=example_html2,
            example_html3=example_html3,
            pygments_css=pygments_css,
        )
        .encode()
    )

    if (here / "docs").is_dir():
        shutil.rmtree(here / "docs")

    # Render main docs
    pdoc.render.configure(
        edit_url_map={
            "activeft": "https://github.com/jonhue/activeft/docs",
        },
        template_directory=here / "templates",
        math=True,
    )
    pdoc.pdoc(
        here / ".." / "activeft",
        output_directory=here / "docs",
    )

    # Add sitemap.xml
    with (here / "sitemap.xml").open("w", newline="\n") as f:
        f.write(
            textwrap.dedent(
                """
        <?xml version="1.0" encoding="utf-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
        """
            ).strip()
        )
        for file in here.glob("**/*.html"):
            if file.name.startswith("_"):
                continue
            filename = str(file.relative_to(here).as_posix()).replace("index.html", "")
            f.write(
                f"""\n<url><loc>https://jonhue.github.io/activeft/{filename}</loc></url>"""
            )
        f.write("""\n</urlset>""")
