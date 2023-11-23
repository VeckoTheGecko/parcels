"""Script to convert `.ipynb` docs to `.md` (omitting cell outputs)"""

from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

PROJECT_ROOT = Path(__file__).parent.parent


def convert_notebooks_to_markdown(input_folder, output_folder):
    # Create a Markdown exporter and exclude cell outputs
    md_exporter = MarkdownExporter()
    md_exporter.exclude_output = True

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Iterate over all .ipynb files in the input folder
    for notebook_path in input_folder.glob('*.ipynb'):
        # Construct markdown file path in output folder
        markdown_path = output_folder / notebook_path.with_suffix('.md').name

        # Read the notebook
        with notebook_path.open('r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)

        # Convert to markdown
        (body, resources) = md_exporter.from_notebook_node(notebook)

        # Write the markdown file
        with markdown_path.open('w', encoding='utf-8') as file:
            file.write(body)

        print(f"Converted {notebook_path.name} to Markdown in {markdown_path}")


if __name__ == "__main__":
    input_path = PROJECT_ROOT / "docs/examples"
    output_path = PROJECT_ROOT / "scripts/md_docs"
    output_path.mkdir(exist_ok=True)

    convert_notebooks_to_markdown(input_path, output_path)
