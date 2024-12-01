import unittest
import os
import time
import cv2
import csv

from src.scripts.face_matcher_models import FaceMatcherModels

def write_summary_to_csv(test_name: str, summary: dict, path: str):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/summary_{test_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test Name', test_name])
        writer.writerow(['Face Matcher Correct', summary['face matcher correct']])
        writer.writerow(['Face Matcher Incorrect', summary['face matcher incorrect']])
        writer.writerow(['Execution Time (seconds)', summary['time']])
        writer.writerow([])
        writer.writerow(['Image 1', 'Image 2', 'Coincidence', 'Distance'])
        for id_image, user_image, coincidence, distance in zip(summary['face1_image'], summary['face2_image'], summary['coincidence'], summary['distance']):
            writer.writerow([id_image, user_image, coincidence, distance])
        writer.writerow([])
        writer.writerow(['Mean Distance', sum(summary['distance']) / len(summary['distance'])])

def image_extension(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def image_from_path(directory):
    directory_contents = os.listdir(directory)
    valid_filenames = [fn for fn in directory_contents if image_extension(fn)]
    return os.path.abspath(os.path.join(directory, valid_filenames[0])) if valid_filenames else None

class TestFaceMatcher(unittest.TestCase):
    def setUp(self):
        self.face_matcher_model = FaceMatcherModels()

    def test_face_matcher_face_recognition_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_face_recognition_model(face1_image,
                                                                                                 face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('face_recognition_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_vgg_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_vgg_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('vgg_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_facenet_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_facenet_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('facenet_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_facenet512_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_facenet512_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('facenet512_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_openface_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_openface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('openface_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_deepface_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_deepface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('deepface_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_deepID_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_deepid_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('deepID_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_arcface_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_arcface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('arcface_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_sface_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_sface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('sface_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_dlib_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_dlib_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('dlib_model_test_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_ghostfacenet_model_matcher_images(self):
        face1_input_folder = 'E:/image_test/similar/faces_1/'
        face2_input_folder = 'E:/image_test/similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_ghostfacenet_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher correct'] += 1
            else:
                summary['face matcher incorrect'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('ghostfacenet_model_test_matcher_faces', summary, 'face_matcher_result')

    ###############################################################################################################################
    # not matcher faces
    def test_face_matcher_face_recognition_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_face_recognition_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('face_recognition_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_vgg_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_vgg_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('vgg_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_facenet_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_facenet_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('facenet_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_facenet512_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_facenet512_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('facenet512_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_openface_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_openface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('openface_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_deepface_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_deepface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('deepface_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_deepID_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_deepid_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('deepID_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_arcface_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_arcface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('arcface_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_sface_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_sface_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('sface_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_dlib_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_dlib_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('dlib_model_test_not_matcher_faces', summary, 'face_matcher_result')

    def test_face_matcher_ghostfacenet_model_not_matcher_images(self):
        face1_input_folder = 'E:/image_test/not_similar/faces_1/'
        face2_input_folder = 'E:/image_test/not_similar/faces_2/'
        summary = {'face matcher correct': 0, 'face matcher incorrect': 0, 'time': 0, 'face1_image': [],
                   'face2_image': [], 'coincidence': [], 'distance': []}
        start_time = time.time()
        face1_images = [os.path.join(face1_input_folder, f) for f in os.listdir(face1_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
        face2_images = [os.path.join(face2_input_folder, f) for f in os.listdir(face2_input_folder) if
                        f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]

        for face1_image_path, face2_image_path in zip(face1_images, face2_images):
            face1_image = cv2.imread(face1_image_path)
            face2_image = cv2.imread(face2_image_path)
            coincidence, distance = self.face_matcher_model.face_matching_ghostfacenet_model(face1_image, face2_image)
            if coincidence:
                summary['face matcher incorrect'] += 1
            else:
                summary['face matcher correct'] += 1
            summary['face1_image'].append(os.path.basename(face1_image_path))
            summary['face2_image'].append(os.path.basename(face2_image_path))
            summary['coincidence'].append(coincidence)
            summary['distance'].append(distance)

        end_time = time.time()
        execution_time = end_time - start_time
        summary['time'] = round(execution_time, 3)

        print(f'Results: {summary}')
        write_summary_to_csv('ghostfacenet_model_test_not_matcher_faces', summary, 'face_matcher_result')