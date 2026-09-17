import argparse
import mimetypes
from lib.llm_client import rewrite_query_based_on_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Image Describer")
    parser.add_argument("--image", required=True, help="Path to a local image to describe")
    parser.add_argument("--query", required=True, help="Text query to re-write based on image data")

    args = parser.parse_args()
    query = args.query
    image_path = args.image
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    print(f"rewriting query '{query}' for image {image_path} of type {mime}")
    image_data = read_filebytes(image_path)
    rewritten_query = rewrite_query_based_on_image(mime, image_data, query)
    print(f"Rewritten query: {rewritten_query}")


def read_filebytes(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    main()