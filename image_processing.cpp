#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
    if (argc < 12) {
        cout << "Usage: " << argv[0] << " <image_path> <upsampling_factor_X> <upsampling_factor_Y> <upsampling_factor_Z> <is2D> <melt> <snow_threshold> <do_x_correction> <do_y_correction> <do_z_correction> <try_gpu>" << endl;
        return -1;
    }

    string image_path = argv[1];
    int upsampling_factor_X = stoi(argv[2]);
    int upsampling_factor_Y = stoi(argv[3]);
    int upsampling_factor_Z = stoi(argv[4]);
    bool is2D = static_cast<bool>(stoi(argv[5]));
    bool melt = static_cast<bool>(stoi(argv[6]));
    float snow_threshold = stof(argv[7]);
    bool do_x_correction = static_cast<bool>(stoi(argv[8]));
    bool do_y_correction = static_cast<bool>(stoi(argv[9]));
    bool do_z_correction = static_cast<bool>(stoi(argv[10]));
    bool try_gpu = static_cast<bool>(stoi(argv[11]));

    // Load the image
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Could not open or find the image: " << image_path << endl;
        return -1;
    }

    // Perform the processing based on the parameters
    // For demonstration, let's just resize the image using the provided upsampling factors

    Mat upscaled_image;
    resize(image, upscaled_image, Size(), upsampling_factor_X, upsampling_factor_Y, INTER_CUBIC);

    // Save the processed image
    string output_path = "processed_image.png";
    imwrite(output_path, upscaled_image);

    cout << "Image processed and saved to " << output_path << endl;

    return 0;
}


void calculate_mean(const Mat& src, Mat& dst, int start, int end, int axis) {
    if (axis == 0) {
        for (int i = 0; i < src.cols; ++i) {
            Scalar mean_val = mean(src(Range(start, end), Range(i, i+1)));
            dst.at<float>(start, i) = mean_val[0];
        }
    } else if (axis == 1) {
        for (int i = 0; i < src.rows; ++i) {
            Scalar mean_val = mean(src(Range(i, i+1), Range(start, end)));
            dst.at<float>(i, start) = mean_val[0];
        }
    } else {
        throw invalid_argument("Axis out of range");
    }
}

void remapping1D(Mat& remapped_image, const Mat& zoomed_image, float upsampling_factor) {
    int dim = remapped_image.rows;
    int dim_upsampled = zoomed_image.rows;
    float sum_correction_factor = 0;

    for (int row = 0; row < dim; ++row) {
        float correction_factor = 1 / (M_PI * sqrt(-1 * (row + 0.5) * (row + 0.5 - dim)));
        sum_correction_factor += correction_factor;
        int upsampled_row = static_cast<int>(round(dim_upsampled * sum_correction_factor));
        int bins = static_cast<int>(round(dim * upsampling_factor * correction_factor));
        
        for (int pixels = 0; pixels < remapped_image.cols; ++pixels) {
            calculate_mean(zoomed_image, remapped_image, upsampled_row, upsampled_row + bins, 0);
        }
    }
}

Mat remapping2D(const Mat& remapped_image, float upsampling_factor) {
    Mat zoomed_image;
    resize(remapped_image, zoomed_image, Size(), upsampling_factor, upsampling_factor, INTER_LINEAR);

    Mat result(remapped_image.size(), CV_32F);
    remapping1D(result, zoomed_image, upsampling_factor);

    return result;
}

Mat process_2D(Mat remapped_image, vector<int> upsampling_factors, bool do_x_correction, bool do_y_correction) {
    if (do_y_correction) {
        remapped_image = remapping2D(remapped_image, upsampling_factors[1]);
    }
    if (do_x_correction) {
        remapped_image = remapped_image.t();
        remapped_image = remapping2D(remapped_image, upsampling_factors[0]);
        remapped_image = remapped_image.t();
    }
    return remapped_image;
}

void save_image(const Mat& remapped_image, const string& filename) {
    imwrite(filename, remapped_image);
}
