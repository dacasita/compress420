#include "myfunctions.h"

cv::Mat ConvertRGBtoYCbCr(const cv::Mat& src, const std::vector<std::vector<float>>& transformationMatrix) {
// Create a new Mat with the same dimensions as the source but with CV_8UC3 type
cv::Mat dst(src.rows, src.cols, CV_8UC3);

for (int y = 0; y < src.rows; ++y) {
for (int x = 0; x < src.cols; ++x) {
// Extract individual R, G, B values
cv::Vec3b rgb = src.at<cv::Vec3b>(y, x);
float B = rgb[2];
float G = rgb[1];
float R = rgb[0];

// Apply the transformation matrix to convert RGB to YCbCr
uchar Y = static_cast<uchar>(transformationMatrix[0][0] * R + transformationMatrix[0][1] * G + transformationMatrix[0][2] * B);
uchar Cb = static_cast<uchar>(128 + transformationMatrix[1][0] * R + transformationMatrix[1][1] * G + transformationMatrix[1][2] * B);
uchar Cr = static_cast<uchar>(128 + transformationMatrix[2][0] * R + transformationMatrix[2][1] * G + transformationMatrix[2][2] * B);

// Assign the YCbCr values to the destination image
dst.at<cv::Vec3b>(y, x) = cv::Vec3b(Y, Cb, Cr);
}
}
return dst;
}

// RGBtoYCbCr420
cv::Mat RGBtoYCbCr420(const cv::Mat& src, const std::vector<std::vector<float>>& transformationMatrix) {
// Create a new Mat with the same dimensions as the source but with CV_8UC3 type
cv::Mat dst(src.rows, src.cols, CV_8UC3, cv::Scalar(0));

for (int y = 0; y < src.rows; ++y) 
{
    for (int x = 0; x < src.cols; ++x) 
    {
    // Extract individual R, G, B values
    cv::Vec3b rgb = src.at<cv::Vec3b>(y, x);
    float B = rgb[2];
    float G = rgb[1];
    float R = rgb[0];

    // Apply the transformation matrix to convert RGB to YCbCr
    uchar Y = static_cast<uchar>(transformationMatrix[0][0] * R + transformationMatrix[0][1] * G + transformationMatrix[0][2] * B);
    uchar Cb;
    uchar Cr;
    if( (y & 1) && ( x & 1))
    {
        Cb = static_cast<uchar>(128 + transformationMatrix[1][0] * R + transformationMatrix[1][1] * G + transformationMatrix[1][2] * B);
        Cr = static_cast<uchar>(128 + transformationMatrix[2][0] * R + transformationMatrix[2][1] * G + transformationMatrix[2][2] * B);
    }

    // Assign the YCbCr values to the destination image
    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(Y, Cb, Cr);
    }
}
return dst;
}
// interpolation
cv::Mat interpolation(const cv::Mat& src) 
{
    // Create a new Mat with the same dimensions as the source but with CV_8UC3 type
    cv::Mat dst(src.rows, src.cols, CV_8UC3, cv::Scalar(0));

    for (int y = 0; y < src.rows; ++y) 
    {
        for (int x = 0; x < src.cols; ++x) 
        {
            // Extract individual R, G, B values
            cv::Vec3b rgb = src.at<cv::Vec3b>(y, x);
            float Cr_old = rgb[2];
            float Cb_old = rgb[1];
            float Y_old = rgb[0];

            uchar Cb;
            uchar Cr;
            if( (y & 1) && ( x & 1 ))
            {
                Cb = Cb_old;
                Cr = Cr_old;
            }else if (y & 1 == 0)
                {
                    cv::Vec3b yCbCr;
                    if(x & 1)
                    {
                        yCbCr = src.at<cv::Vec3b>(y+1, x);
                    }else
                    {
                        yCbCr = src.at<cv::Vec3b>(y+1, x+1);
                    }
                    Cb = yCbCr[1];
                    Cr = yCbCr[2];
                }
                else
                {
                    cv::Vec3b yCbCr = src.at<cv::Vec3b>(y, x+1);
                    Cb = yCbCr[1];
                    Cr = yCbCr[2];
                }

            // Assign the YCbCr values to the destination image
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(Y_old, Cb, Cr);
        }
    }
    return dst;
}

cv::Mat ConvertYCbCrToRGB(const cv::Mat& ycbcr_image, const std::vector<std::vector<float>>& inverseMatrix) {
// Check if the inverse matrix is 3x3
if (inverseMatrix.size() != 3 || inverseMatrix[0].size() != 3 ||
inverseMatrix[1].size() != 3 || inverseMatrix[2].size() != 3) {
throw std::runtime_error("Inverse matrix must be 3x3");
}

// Create an empty RGB image with the same dimensions as the YCbCr image
cv::Mat rgb_image(ycbcr_image.rows, ycbcr_image.cols, CV_8UC3);

for (int y = 0; y < ycbcr_image.rows; ++y) {
for (int x = 0; x < ycbcr_image.cols; ++x) {
// Get the YCbCr pixel
cv::Vec3b ycbcr_pixel = ycbcr_image.at<cv::Vec3b>(y, x);

//ycbcr_pixel[1] -= 128;
//ycbcr_pixel[2] -= 128;

double Cb = ycbcr_pixel[1] - 128;
double Cr = ycbcr_pixel[2] - 128;
// Apply the inverse matrix to get the RGB values
float R = inverseMatrix[0][0] * ycbcr_pixel[0] + inverseMatrix[0][1] * Cb + inverseMatrix[0][2] * Cr;
float G = inverseMatrix[1][0] * ycbcr_pixel[0] + inverseMatrix[1][1] * Cb + inverseMatrix[1][2] * Cr;
float B = inverseMatrix[2][0] * ycbcr_pixel[0] + inverseMatrix[2][1] * Cb + inverseMatrix[2][2] * Cr;

// Clip the values to be in the 0 to 255 range
uchar r = static_cast<uchar>(std::max(0.0f, std::min(R, 255.0f)));
uchar g = static_cast<uchar>(std::max(0.0f, std::min(G, 255.0f)));
uchar b = static_cast<uchar>(std::max(0.0f, std::min(B, 255.0f)));

// Assign the RGB values to the destination image
rgb_image.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
}
}

return rgb_image;
}


int main() {
// Load the image
cv::Mat image = cv::imread("Lenna.jpg");

// Check for failure
if (image.empty()) {
std::cout << "Could not open or find the image" << std::endl;
return -1;
}

// Define the transformation matrix for RGB to YCbCr
const std::vector<std::vector<float>> transformationMatrix = {
{0.299, 0.587, 0.114},
{-0.168736, -0.331264, 0.5},
{0.5, -0.418688, -0.081312}
};

// Convert the RGB image to YCbCr
cv::Mat image_YCbCr = RGBtoYCbCr420(image, transformationMatrix);
cv::Mat afterInterpolation = interpolation(image_YCbCr); 

// Define the inverse matrix for YCbCr to RGB conversion
const std::vector<std::vector<float>> ycbcrToRgbMatrix = {
{1, -0.00093, 1.401687},
{1, -0.3437, -0.71417},
{1, 1.77216, 0.00099}
};

//cv::Mat inverseMatrix = cv::Mat::zeros(3, 3, CV_32F);
//cv::invert(transformationMatrix, inverseMatrix);

// Convert the YCbCr image to RGB
cv::Mat rgb_image = ConvertYCbCrToRGB(afterInterpolation, ycbcrToRgbMatrix);

// Display the images
cv::imshow("Original Image", image);
//cv::imshow("Compressed Image", compressed);
cv::imshow("Recovered Image", rgb_image);

// Wait for any keystroke in the window
cv::waitKey(0);

return 0;
}