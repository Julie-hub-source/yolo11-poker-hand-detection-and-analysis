from ultralytics import YOLO
import argparse
import cv2
import os

def detect_cards(image_path, weights_path, conf=0.5, save_output=True):
    '''
    Detects cards in an image using YOLO11 model and returns the unique cards.

    Args:
        image_path (str): Path to the image file.
        weights_path (str): Path to the YOLO11 weights file.
        conf (float): Confidence threshold for the detection.
        save_output (bool): Whether to save the annotated image.

    Returns:
        list: List of unique cards detected in the image, sorted left to right.
    '''
    model = YOLO(weights_path, task='detect')
    results = model.predict(image_path, save=False, conf=conf)[0]
    
    cards = []
    card_names = []
    summary = results.summary()

    for card in summary:
        if card['confidence'] >= conf:
            name = card['name']
            if name not in card_names:
                card_names.append(name)
                left = min(card['box']['x1'], card['box']['x2'])
                cards.append((left, name))

    cards.sort(key=lambda x: x[0])
    detected_names = [card[1] for card in cards]

    if save_output:
        annotated = results.plot()  # BGR image with boxes
        out_path = 'output_result.jpg'
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image to {out_path}")

    return detected_names

def decode_cards(cards):
    # (Same dictionary as your original decode_cards function)
    card_names = {
        '2C': '2 of Clubs', '3C': '3 of Clubs', '4C': '4 of Clubs', '5C': '5 of Clubs',
        '6C': '6 of Clubs', '7C': '7 of Clubs', '8C': '8 of Clubs', '9C': '9 of Clubs',
        '10C': '10 of Clubs', 'JC': 'Jack of Clubs', 'QC': 'Queen of Clubs', 'KC': 'King of Clubs',
        '2D': '2 of Diamonds', '3D': '3 of Diamonds', '4D': '4 of Diamonds', '5D': '5 of Diamonds',
        '6D': '6 of Diamonds', '7D': '7 of Diamonds', '8D': '8 of Diamonds', '9D': '9 of Diamonds',
        '10D': '10 of Diamonds', 'JD': 'Jack of Diamonds', 'QD': 'Queen of Diamonds', 'KD': 'King of Diamonds',
        '2H': '2 of Hearts', '3H': '3 of Hearts', '4H': '4 of Hearts', '5H': '5 of Hearts',
        '6H': '6 of Hearts', '7H': '7 of Hearts', '8H': '8 of Hearts', '9H': '9 of Hearts',
        '10H': '10 of Hearts', 'JH': 'Jack of Hearts', 'QH': 'Queen of Hearts', 'KH': 'King of Hearts',
        '2S': '2 of Spades', '3S': '3 of Spades', '4S': '4 of Spades', '5S': '5 of Spades',
        '6S': '6 of Spades', '7S': '7 of Spades', '8S': '8 of Spades', '9S': '9 of Spades',
        '10S': '10 of Spades', 'JS': 'Jack of Spades', 'QS': 'Queen of Spades', 'KS': 'King of Spades'
    }

    return [card_names.get(card, card) for card in cards]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--weights', type=str, default='weights/poker_best.pt', help="Path to YOLOv11 weights")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--save_output', action='store_true', help="Save annotated result image")
    args = parser.parse_args()

    cards = detect_cards(args.img_path, args.weights, args.conf, args.save_output)
    decoded = decode_cards(cards)
    print("Detected Cards:", ", ".join(decoded))
