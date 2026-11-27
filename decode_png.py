#!/usr/bin/env python3
"""Decode QR codes in a PNG, or test the latest saved camera capture."""
import argparse
from pathlib import Path
import cv2
import zxingcpp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', help='Optional path to an image or capture folder')
    args = parser.parse_args()
    if args.image:
        target = Path(args.image).expanduser()
        if not target.exists():
            parser.error(f'Path does not exist: {target}')
    else:
        root = Path(__file__).resolve().parent / 'qr_captures'
        frames = list(root.glob('*/frame.png'))
        if not frames:
            parser.error('No saved frames found. Run capture_qr_once.py first.')
        target = max(frames, key=lambda p: p.stat().st_mtime).parent
    files = [target] if target.is_file() else [target / name for name in
        ['frame.png', 'crop.png', 'rectified.png'] if (target / name).is_file()]
    if not files:
        parser.error(f'No frame.png, crop.png, or rectified.png found in {target}')
    successes = set()
    for path in files:
        print(f'\nImage: {path}')
        image = cv2.imread(str(path))
        if image is None:
            print('Could not load image.')
            continue
        try:
            text = cv2.QRCodeDetector().detectAndDecode(image)[0]
            print('OpenCV:', repr(text) if text else 'Not decoded')
            if text:
                successes.add(text)
        except Exception as exc:
            print('OpenCV error:', exc)
        try:
            codes = zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.QRCode)
            texts = [code.text for code in codes]
            print('ZXing:', repr(texts) if texts else 'Not decoded')
            successes.update(texts)
        except Exception as exc:
            print('ZXing error:', exc)
    print('\nDecoded contents:' if successes else '\nNo QR contents decoded from these images.')
    for text in sorted(successes):
        print(repr(text))


if __name__ == '__main__':
    main()
