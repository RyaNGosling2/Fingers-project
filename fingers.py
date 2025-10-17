import cv2
import numpy as np

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary)
    return skeleton

def skeletonize(img):
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton

def extract_features(skeleton):
    features = []
    rows, cols = skeleton.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skeleton[i, j] == 255:
                neighbors = 0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if x == 0 and y == 0:
                            continue
                        if skeleton[i + x, j + y] == 255:
                            neighbors += 1
                if neighbors == 1 or neighbors == 3:
                    features.append({
                        'x': j, 
                        'y': i, 
                        'type': 'ending' if neighbors == 1 else 'bifurcation'})
    return features

def find_best_matches(sample_features, test_features, max_distance=15):
    matches = []
    used_test_features = set() 
    
    for i, sample_feat in enumerate(sample_features):
        best_match_idx = -1
        best_distance = float('inf')
        for j, test_feat in enumerate(test_features):
            if j in used_test_features:
                continue
            if sample_feat['type'] != test_feat['type']:
                continue
            distance = np.sqrt((sample_feat['x'] - test_feat['x'])**2 + 
                             (sample_feat['y'] - test_feat['y'])**2)
            if distance < max_distance and distance < best_distance:
                best_distance = distance
                best_match_idx = j
        if best_match_idx != -1:
            matches.append({
                'sample_idx': i,
                'test_idx': best_match_idx,
                'distance': best_distance
            })
            used_test_features.add(best_match_idx)
    return matches

def compare_fingerprints(sample_image_path, test_image_path, threshold):
    print(f"Загрузка образца: {sample_image_path}")
    sample_skeleton = process_image(sample_image_path)
    sample_features = extract_features(sample_skeleton)
    print(f"Сравнение с: {test_image_path}")
    test_skeleton = process_image(test_image_path)
    test_features = extract_features(test_skeleton)
    matches = find_best_matches(sample_features, test_features)
    total_sample_features = len(sample_features)
    total_test_features = len(test_features)
    total_matches = len(matches)
    min_features = min(total_sample_features, total_test_features)
    if min_features > 0:
        similarity = total_matches / min_features
    else:
        similarity = 0
    print("\n--- СТАТИСТИКА СРАВНЕНИЯ ---")
    print(f"Особенностей в образце: {total_sample_features}")
    print(f"Особенностей в тесте:   {total_test_features}")
    print(f"Найдено совпадений:     {total_matches}")
    print(f"Схожесть:               {similarity:.2f} ({similarity*100:.1f}%)")
    if total_matches > 0:
        avg_distance = sum(m['distance'] for m in matches) / total_matches
        print(f"Среднее расстояние:     {avg_distance:.2f} пикселей (3-8 хорошо, 8-15 нормально, 15+ плохо)")
    print(f"Порог схожести:         {threshold} ({threshold*100:.1f}%)")
    if similarity >= threshold:
        print("\nРезультат: ОТПЕЧАТКИ СОВПАДАЮТ")
        return True
    else:
        print("\nРезультат: отпечатки НЕ совпадают")
        return False

sample = "110_1.tif"
test = "109_2.tif"
threshold = 0.9
result = compare_fingerprints(sample, test, threshold)