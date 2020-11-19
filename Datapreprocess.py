import os
import html
import pickle
import numpy as np
import xml.etree.cElementTree as ElementTree


def remove_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = np.sqrt(np.sum(np.square(p1 - p2))) + np.sqrt(np.sum(np.square(p2 - p3)))

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
	for root, directories, files in os.walk(./Dataset):
		for file in files:
			f_name, f_ext = os.splittext(file) # Splitting the file in its name and its extension
			if f_ext == '.xml': # Checking for xml extension

				# Parsing the xml file
				xml = ElementTree.parse(os.path.join(root, file)).getroot()
				transcription = xml.findall('Transcription')

				# Escaping the data which doesn't have Transcription node
				if not transcription:
					continue

				# Getting the texts for xml file and its stokes points  
				texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]

                strokes = []
                mid_points = []

                for point in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in point])
                    pts[-1, 2] = 1

                    # Removing the middle points between two separated points
                    pts = remove_middle(pts)
                    if len(pts) == 0:
                        continue



if __name__ == "__main__":
	main()