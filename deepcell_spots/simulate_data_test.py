import random
import numpy as np

from tensorflow.python.platform import test

from simulate_data import *

class TestSimulateData(test.TestCase):
    def test_sim_gt_clusters(self):
        num_clusters = 1000
        tp_ratio = 0.5
        gt = sim_gt_clusters(num_clusters,tp_ratio)

        self.assertEqual(len(gt),num_clusters)
        self.assertEqual(np.round(sum(gt)/len(gt),1),tp_ratio)

        # test TP ratio greater than 1
        tp_ratio = 1.1
        with self.assertRaises(AssertionError):
            gt = sim_gt_clusters(num_clusters,tp_ratio)

        # test negative TP ratio
        tp_ratio = -0.1
        with self.assertRaises(AssertionError):
            gt = sim_gt_clusters(num_clusters,tp_ratio)

    def test_sim_detections(self):
        gt_tp = np.ones(1000)
        gt_fp = np.zeros(1000)
        tpr = 0.9
        fpr = 0.1
        det_list_tp = sim_detections(gt_tp, tpr, fpr)
        det_list_fp = sim_detections(gt_fp, tpr, fpr)

        self.assertEqual(len(gt_tp),len(det_list_tp))
        self.assertEqual(len(gt_fp),len(det_list_fp))
        self.assertEqual(np.round(sum(det_list_tp)/len(det_list_tp),1),tpr)
        self.assertEqual(np.round(sum(det_list_fp)/len(det_list_fp),1),fpr)

        # test TPR greater than 1
        tpr = 1.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            det_list_tp = sim_detections(gt_tp, tpr, fpr)

        # test negative TPR
        tpr = -0.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            det_list_tp = sim_detections(gt_tp, tpr, fpr)

    def test_sim_annotators(self):
        gt_tp = np.ones(1000)
        gt_fp = np.zeros(1000)
        tpr_list = np.array([0.9,0.8,0.7]) # to show that either list or np array works
        fpr_list = [0.1,0.2,0.3]
        data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)
        data_array_fp = sim_annotators(gt_fp,tpr_list,fpr_list)

        self.assertEqual(np.shape(data_array_tp),(len(gt_tp),len(tpr_list)))
        self.assertEqual(np.shape(data_array_fp),(len(gt_fp),len(tpr_list)))
        for i in range(len(tpr_list)):
            self.assertEqual(np.round(sum(data_array_tp[:,i])/len(data_array_tp[:,i]),1),tpr_list[i])
            self.assertEqual(np.round(sum(data_array_fp[:,i])/len(data_array_fp[:,i]),1),fpr_list[i])

        # test incorrect type for tpr and fpr
        tpr = 1.1
        fpr = 0.1
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr,fpr)

        # test different lengths for tpr_list and fpr_list
        tpr_list = [0.9]*4
        fpr_list = [0.1]*3
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)

        # test TPR greater than 1
        tpr_list = [0.9,1,1.1]
        fpr_list = [0.1]*3
        with self.assertRaises(AssertionError):
            data_array_tp = sim_annotators(gt_tp,tpr_list,fpr_list)

    def test_percent_correct(self):
        # test all wrong detections
        num_detections = 10
        gt = np.ones(num_detections)
        data_array = np.array([np.zeros(num_detections),np.ones(num_detections)]).T
        perc_corr = percent_correct(gt, data_array)
        self.assertEqual(perc_corr, 0)

        # test all correct detections
        num_detections = 10
        gt = np.zeros(num_detections)
        data_array = np.array([np.zeros(num_detections),np.ones(num_detections)]).T
        perc_corr = percent_correct(gt, data_array)
        self.assertEqual(perc_corr, 1)

        # test all correct detections
        num_detections = 10
        gt = np.ones(num_detections)
        data_array = np.array([np.ones(num_detections),np.zeros(num_detections)]).T
        perc_corr = percent_correct(gt, data_array)
        self.assertEqual(perc_corr, 1)

        # test half correct detections
        data_array = np.array([np.concatenate((np.zeros(int(num_detections/2)),np.ones(int(num_detections/2)))),
                                np.concatenate((np.ones(int(num_detections/2)),np.zeros(int(num_detections/2))))]).T
        perc_corr = percent_correct(gt, data_array)
        self.assertEqual(perc_corr, 0.5)

        data_array = np.ones(num_detections-1)
        with self.assertRaises(AssertionError):
            perc_corr = percent_correct(gt, data_array)

    def test_is_in_image(self):
        x = 0
        y = 0
        a = 10
        L = 100
        in_image = is_in_image(x,y,a,L)
        self.assertEqual(in_image, True)

        # test square bigger than image
        a = 100
        L = 10
        in_image = is_in_image(x,y,a,L)
        self.assertEqual(in_image, False)

        # test negative coordinates for square corner
        x = -1
        y = -1
        a = 10
        L = 100
        in_image = is_in_image(x,y,a,L)
        self.assertEqual(in_image, False)

    def test_is_overlapping(self):
        # test squares compared are exactly overlapping
        x_list = [0]
        y_list = [0]
        a_list = [10]
        x = 0
        y = 0
        a = 10
        overlapping = is_overlapping(x_list,y_list,a_list,x,y,a)
        self.assertEqual(overlapping, True)

        # test squares are not overlapping
        x_list = [11]
        y_list = [11]
        a_list = [10]
        x = 0
        y = 0
        a = 10
        overlapping = is_overlapping(x_list,y_list,a_list,x,y,a)
        self.assertEqual(overlapping, False)

        # test one square overlapping, other not
        x_list = [0,11]
        y_list = [0,11]
        a_list = [10,10]
        x = 0
        y = 0
        a = 10
        overlapping = is_overlapping(x_list,y_list,a_list,x,y,a)
        self.assertEqual(overlapping, True)

        # test lists of different lengths
        x_list = [0]
        with self.assertRaises(AssertionError):
            overlapping = is_overlapping(x_list,y_list,a_list,x,y,a)

    def test_add_gaussian_noise(self):
        img_w, img_h = 30, 30
        X = np.random.random((img_w, img_h))
        noisy = add_gaussian_noise(X, m=3, s=3)
        self.assertEqual(np.shape(X), np.shape(noisy))
        self.assertGreaterEqual(noisy.all(), X.all())

    def test_gaussian_spot_image_generator(self):
        L = 30
        N_min = 1
        N_max = 5
        g = gaussian_spot_image_generator(L=L,N_min=N_min,N_max=N_max,
                                            sigma_mean=1,sigma_std=0.5,
                                            A_mean=1,A_std=0.5,
                                            segmask=True,yield_pos=True)

        img, label, x_list, y_list, bboxes = next(g)
        self.assertEqual(np.shape(img),(L,L))
        self.assertEqual(np.shape(label),(L,L))
        self.assertEqual(len(x_list),len(y_list),len(bboxes))
        self.assertGreaterEqual(len(x_list),N_min)
        self.assertLessEqual(len(x_list),N_max)

        L = 30
        N_min = 1
        N_max = 5
        g = gaussian_spot_image_generator(L=L,N_min=N_min,N_max=N_max,
                                            sigma_mean=1,sigma_std=0.5,
                                            A_mean=1,A_std=0.5,
                                            segmask=False,yield_pos=True)

        img, label, x_list, y_list, bboxes = next(g)
        self.assertEqual(np.shape(img),(L,L))
        self.assertEqual(np.shape(label),(L,L))
        self.assertEqual(len(x_list),len(y_list),len(bboxes))
        self.assertGreaterEqual(len(x_list),N_min)
        self.assertLessEqual(len(x_list),N_max)

        L = 30
        N_min = 1
        N_max = 5
        g = gaussian_spot_image_generator(L=L,N_min=N_min,N_max=N_max,
                                            sigma_mean=1,sigma_std=0.5,
                                            A_mean=1,A_std=0.5,
                                            segmask=False,yield_pos=False)

        img, label = next(g)
        self.assertEqual(np.shape(img),(L,L))
        self.assertEqual(np.shape(label),(L,L))
test.main()