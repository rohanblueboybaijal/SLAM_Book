#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void findFeatureMatches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches);

void poseEstimation2D2D(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch> matches, cv::Mat &R, cv::Mat &t);

cv::Point2d pixel2Cam(const cv::Point2d &p, const cv::Mat &K);

int main(int argc, char **argv) {
    if(argc != 3) {
        std::cout<<"Usage : pose_estimation_2d2d img1 img2" <<std::endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_ANYCOLOR);

    assert(img_1.data && img_2.data && "Cannot load images");

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    findFeatureMatches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout<<"We get "<<matches.size()<<" set of features"<<std::endl;

    // Estimating motion between 2 frames
    cv::Mat R, t;
    poseEstimation2D2D(keypoints_1, keypoints_2, matches, R, t);

    // Check E = t^R*scale
    cv::Mat t_x = (cv::Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1,0), t.at<double>(2,0), 0, -t.at<double>(0,0),
    -t.at<double>(1,0), t.at<double>(0,0), 0);

    std::cout<<"t^R = "<<std::endl << t_x * R <<std::endl;

    // Check epipolar constraints
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(cv::DMatch m : matches) {
        cv::Point2d pt1 = pixel2Cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2Cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout<<"epipolar constraint = "<< d <<std::endl;
    }
    return 0;
}

void findFeatureMatches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches) {
    cv::Mat descriptors_1, descriptors_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 1000, max_dist = 0;

    for(int i=0; i<descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    for(int i=0; i<descriptors_1.rows; i++) {
        if(match[i].distance <= std::max(2*min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2Cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d(((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0)), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}


void poseEstimation2D2D(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch> matches, cv::Mat &R, cv::Mat &t) {
    // Camera Intrinsics, TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2f> points_1;
    std::vector<cv::Point2f> points_2;

    for(int i=0; i< matches.size(); i++) {
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // Calculate the fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points_1, points_2, CV_FM_8POINT);
    std::cout<<"fundamental_matrix is "<<std::endl<<fundamental_matrix<<std::endl;

    // Calculate essential matrix
    cv::Point2d principal_point(325.1, 249.7); // camera principal point calibrated in TUM dataset
    double focal_length = 521;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points_1, points_2, focal_length, principal_point);
    std::cout<<"essential_matrix is "<<std::endl<<essential_matrix<<std::endl;

    // Calculate the homography matrix
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points_1, points_2, cv::RANSAC, 3);
    std::cout<<"homography_matrix is "<<std::endl<<homography_matrix<<std::endl;

    // Recover rotation and translation
    cv::recoverPose(essential_matrix, points_1, points_2, R, t, focal_length, principal_point);

    std::cout<<"R is "<<std::endl<<R<<std::endl;
    std::cout<<"t is "<<std::endl<<t<<std::endl;
}