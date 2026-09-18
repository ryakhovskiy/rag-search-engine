import argparse
from lib.multimodal_search import verify_image_embedding
from lib.multimodal_search import search_using_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimidal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    vie_parser = subparsers.add_parser("verify_image_embedding", help="Verifies if the model is properly loaded and image embedding can be generated")
    vie_parser.add_argument("image", type=str, help="Path to the image to generate embedding for")

    img_search = subparsers.add_parser("image_search", help="Search using images")
    img_search.add_argument("image", type=str, help="Path to the image for the search query")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            image = args.image
            verify_image_embedding(image)
        case "image_search":
            image = args.image
            search_using_image(image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()