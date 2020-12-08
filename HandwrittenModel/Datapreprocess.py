import os
import html
import pickle
import numpy as np
import xml.etree.cElementTree as ElementTree


def seperate(pts):
    seperated_points = []
    for i in range(0, len(pts) - 1):
        # Filtering the points at a distance greater than 600
        if np.sqrt(np.sum(np.square(pts[i] - pts[i+1]))) > 600:
            seperated_points += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seperated_points, seperated_points + [len(pts)])]


def remove_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = np.sqrt(np.sum(np.square(p1 - p2))) + \
            np.sqrt(np.sum(np.square(p2 - p3)))

        # Clearing the middle point between two points having distance greater than 1500
        if dist > 1500:
            to_remove.add(i)

    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def main():
    data = []
    charset = set()

    # Reading the xml files in our Dataset folder
    for root, _, files in os.walk('./HandwrittenModel/Dataset'):
        for file in files:
            # Splitting the file in its name and its extension
            f_name, f_ext = os.path.splitext(file)
            if f_ext == '.xml':  # Checking for xml extension

                # Parsing the xml file
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')

                # Escaping the data which doesn't have Transcription node
                if not transcription:
                    continue

                # Getting the texts for xml file and its stokes points
                texts = [html.unescape(s.get('text'))
                         for s in transcription[0].findall('TextLine')]
                stroke_set = xml.findall('StrokeSet')[0].findall('Stroke')

                points = [s.findall('Point') for s in stroke_set]
                strokes = []
                mid_points = []

                for point in points:
                    pts = np.array(
                        [[int(p.get('x')), int(p.get('y')), 0] for p in point])
                    pts[-1, 2] = 1

                    # Removing the middle points between two separated points
                    pts = remove_middle(pts)
                    if len(pts) == 0:
                        continue

                    # Filtering stroke points at a distance

                    seperated_points = seperate(pts)
                    for pss in seperated_points:
                        if len(seperated_points) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[
                            0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[
                            0], min(pss, key=lambda x: x[1])[1]

                        # Adding the strokes points
                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2.,
                                        (ymax + ymin) / 2.]]

                print(f"Processing file : {f_name}")

                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]

                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)

                lines = []

                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]

                # Adding the characterset
                charset |= set(''.join(texts))

                # data is in form of tuples for text and corresponding line strokes
                data += [(texts, lines)]

    translation = {'<NULL>': 0}
    for character in ''.join(sorted(charset)):
        translation[character] = len(translation)

    dataset = []
    labels = []

    # converting the required data in numpy arrays and adding the required lables from translation
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            # Storing the min point for the starting of each text
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]

            # Mapping the translations to the texts as labels
            labels += list(map(lambda x: translation[x], text))

    whole_data = np.concatenate(dataset, axis=0)
    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    try:
        # Trying to make 'data' folder if it doesn't already exists
        os.makedirs('HandwrittenModel/data')
    except FileExistsError:
        pass

    # Saving our processed data into the 'data' folder
    np.save(os.path.join('HandwrittenModel/data', 'dataset'), np.array(dataset))
    np.save(os.path.join('HandwrittenModel/data', 'labels'), np.array(labels))
    with open(os.path.join('HandwrittenModel/data', 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)


if __name__ == "__main__":
    main()
