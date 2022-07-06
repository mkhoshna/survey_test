import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely.geometry
from shapely.geometry import Polygon
from shapely import wkt
import similaritymeasures
import statistics
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

class survey:
    '''
        class survey models a single survey that includes two polygons, each representing a farm.
    '''
    def __init__(self, a, b, verbosity=0):
        self.verbosity = verbosity
        self.polyA = a
        self.polyB = b
        self.intersect = None
        self.intersect_metric = None
        self.centroid_metric = None
        self.frechet_metric = None
        self.org_polyA = None
        self.org_polyB = None
        self.fixed_flag = False

    def init_sample(self):
        self.polyA = Polygon([[1, 1], [1, 3], [3, 3], [3, 1]])
        self.polyB = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])

    def draw(self):
        plt.show(block=True)
        plt.style.use('bmh')

        fig, ax = plt.subplots()
        if self.fixed_flag:
            a = [self.polyA, self.polyB, self.org_polyA, self.org_polyB]
        else:
            a = [self.polyA, self.polyB]
        for o in a:
            x,y = o.exterior.xy
            ax.plot(x, y, linewidth=2.0)
        plt.show()
        return

    def calc_overlap(self):
        return

    def calc_metric(self):
        self.intersect = self.polyA.intersection(self.polyB)
        self.intersect_metric = [self.intersect.area/self.polyA.area, self.intersect.area/self.polyB.area]
        self.centroid_metric = self.polyA.centroid.distance(self.polyB.centroid)

        x_a, y_a = self.polyA.exterior.xy
        x_b, y_b = self.polyB.exterior.xy
        curve_a = np.zeros((len(x_a), 2))
        curve_a[:, 0] = x_a.tolist()
        curve_a[:, 1] = y_a.tolist()
        curve_b = np.zeros((len(x_b), 2))
        curve_b[:, 0] = x_b.tolist()
        curve_b[:, 1] = y_b.tolist()
        self.frechet_metric = similaritymeasures.frechet_dist(curve_a, curve_b)

        return [self.intersect_metric[0], self.intersect_metric[1], self.centroid_metric, self.frechet_metric]

    @staticmethod
    def filter_polygon_points(a: shapely.geometry.Polygon, b: shapely.geometry.Polygon):
        a_filtered_pts = list()
        x_a = a.exterior.coords.xy[0].tolist()
        y_a = a.exterior.coords.xy[1].tolist()
        assert len(x_a) == len(y_a), "ERROR 12: x and y coordinates should have the same length"
        for idx in range(len(x_a)):
            pt = shapely.geometry.Point(x_a[idx], y_a[idx])
            if b.contains(pt) is False:
                a_filtered_pts.append(pt)
        return shapely.geometry.Polygon(a_filtered_pts)

    def fix_overlapping_borders(self):
        if self.polyA.intersects(self.polyB):
            self.org_polyA = self.polyA
            self.org_polyB = self.polyB
            self.fixed_flag = True
            self.polyA = survey.filter_polygon_points(self.org_polyA, self.org_polyB)
            self.polyB = survey.filter_polygon_points(self.org_polyB, self.org_polyA)

    def fix_similar_borders(self):
        updated_pts = list()
        x_a = self.polyA.exterior.coords.xy[0].tolist()
        y_a = self.polyA.exterior.coords.xy[1].tolist()
        x_b = self.polyA.exterior.coords.xy[0].tolist()
        y_b = self.polyA.exterior.coords.xy[1].tolist()
        x_c = statistics.mean([self.polyA.centroid.coords.xy[0][0], self.polyB.centroid.coords.xy[0][0]])
        y_c = statistics.mean([self.polyA.centroid.coords.xy[1][0], self.polyB.centroid.coords.xy[1][0]])
        x_max_a = max(abs(x_a[idx] - x_c) for idx in range(len(x_a)))
        x_max_b = max(abs(x_b[idx] - x_c) for idx in range(len(x_b)))
        x_max = max([x_max_a, x_max_b])
        y_max_a = max(abs(y_a[idx] - y_c) for idx in range(len(y_a)))
        y_max_b = max(abs(y_b[idx] - y_c) for idx in range(len(y_b)))
        y_max = max([y_max_a, y_max_b])
        max_rad = math.sqrt(x_max ** 2 + y_max ** 2)
        x = np.arange(x_c, max_rad+x_c, 0.1*max_rad)
        y = np.ones_like(x) * y_c
        t_angle = np.arange(0, 2*math.pi, 0.1*math.pi/10)
        for t in t_angle:
            c, s = np.cos(t), np.sin(t)
            rotation_matrix = np.array(((c, -s), (s, c)))
            rot_line = rotation_matrix.dot([x, y])
            x_rot_line = rot_line[0]
            x_rot_line = x_rot_line - x_rot_line[0] + x_c
            y_rot_line = rot_line[1]
            y_rot_line = y_rot_line - y_rot_line[0] + y_c
            rotated_line = shapely.geometry.LineString(np.column_stack([x_rot_line, y_rot_line]))
            intersect_a = self.polyA.intersection(rotated_line)
            intersect_b = self.polyB.intersection(rotated_line)
            if intersect_a.type == "LineString" and intersect_b.type == "LineString":
                intersect_a_coords = np.asarray([intersect_a.coords.xy])
                intersect_a_coords_x = statistics.mean(intersect_a_coords[0][0])
                intersect_a_coords_y = statistics.mean(intersect_a_coords[0][1])
                intersect_b_coords = np.asarray([intersect_b.coords.xy])
                intersect_b_coords_x = statistics.mean(intersect_b_coords[0][0])
                intersect_b_coords_y = statistics.mean(intersect_b_coords[0][1])
                x_avg = statistics.mean([intersect_a_coords_x, intersect_b_coords_x])
                y_avg = statistics.mean([intersect_a_coords_y, intersect_b_coords_y])
                pt = [x_avg, y_avg]
                updated_pts.append(pt)

        self.org_polyA = self.polyA
        self.org_polyB = self.polyB
        self.fixed_flag = True
        self.polyA = shapely.geometry.Polygon(updated_pts)
        self.polyB = shapely.geometry.Polygon(updated_pts)



    def fix_borders(self):
        self.fix_overlapping_borders(self)
        self.fix_similar_borders(self)


class survey_set:
    def __init__(self, verbosity=0):
        self.verbosity = verbosity
        self.sset = list()
        self.metrics = list()
        self.classifier = None

    def import_data(self, file_name):
        if self.verbosity > 0:
            print("Importing survey set data from file: {} ...".format(file_name))
        data = pd.read_csv(file_name, delimiter=";")
        data = data.iloc[:, 1:]
        data = data.drop_duplicates()
        polygons_pair_a = data['Pair_a'].tolist()
        polygons_pair_b = data['Pair_b'].tolist()

        assert len(polygons_pair_a)==len(polygons_pair_b), "ERROR 13: Each survey should contain exactly two polygons!"

        if self.verbosity > 0:
            print("Calculating metrics ...")
            if self.verbosity > 1:
                printProgressBar(0, len(polygons_pair_a), prefix='Progress:', suffix='Complete', length=50)

        for idx in range(len(polygons_pair_a)):
            poly_a = wkt.loads(polygons_pair_a[idx])
            poly_b = wkt.loads(polygons_pair_b[idx])
            s = survey(poly_a, poly_b)
            self.metrics.append(s.calc_metric())
            self.sset.append(s)

            if self.verbosity > 1:
                printProgressBar(idx, len(polygons_pair_a), prefix='Progress:', suffix='Complete', length=50)

        if self.verbosity > 0:
            print("Importing survey set data and calculating metrics: DONE!")

    def classify(self):
        self.classifier = survey_classifier(verbosity=self.verbosity)
        self.classifier.train_and_validate(self.metrics)


class survey_classifier:
    def __init__(self, verbosity=0):
        self.model = None
        self.cf_matrix = None
        self.mean_scores = None
        self.verbosity = verbosity

    def train_and_validate(self, data):
        saved_metrics_labels = pd.read_csv('C:\\Users\\Mahta\\PycharmProjects\\TR_assessment\\features_labels.csv')
        features = pd.DataFrame(data)
        labels = saved_metrics_labels['Label']
        self.model = LinearDiscriminantAnalysis()
        self.model.fit(features, labels)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scores = cross_val_score(self.model, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        self.mean_scores = np.mean(scores)
        if self.verbosity > 0:
            print("Model prediction accuracy = {}".format(self.mean_scores))

        predicted_labels = self.model.predict(features)
        self.cf_matrix = confusion_matrix(labels, predicted_labels)
        if self.verbosity > 0:
            print("Model Confusion Matrix= \n {}".format(self.cf_matrix))

        return self.mean_scores, self.cf_matrix
