#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <cstdlib>
#include <string>

void adjustBrightnessContrast(cv::Mat& frame, int brightness, int contrast) {
    double alpha = (contrast + 100) / 100.0;
    int beta = brightness;
    frame.convertTo(frame, CV_8UC3, alpha, beta);
}

cv::Mat adaptiveHistogramEqualization(const cv::Mat& frame) {
    cv::Mat gray, equalized, equalizedBGR;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, equalized);
    cv::cvtColor(equalized, equalizedBGR, cv::COLOR_GRAY2BGR);
    return equalizedBGR;
}

void processVideo(const std::string& file, const std::string& outputDir, bool display) {
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << file << std::endl;
        return;
    }

    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::string outputName = outputDir + "/" + std::filesystem::path(file).stem().string() + "_processed.mp4";

    std::string ffmpegCmd = "ffmpeg -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) + 
                             " -r " + std::to_string(fps) + " -i pipe: -c:v libx264 -preset ultrafast -tune film -pix_fmt yuv420p " + outputName;

    FILE* pipe = popen(ffmpegCmd.c_str(), "w");
    if (!pipe) {
        std::cerr << "Error opening pipe for ffmpeg" << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        adjustBrightnessContrast(frame, 150, 210);
        cv::Mat equalizedFrame = adaptiveHistogramEqualization(frame);
        
        // Write the frame to ffmpeg pipe
        fwrite(equalizedFrame.data, sizeof(unsigned char), equalizedFrame.total() * equalizedFrame.elemSize(), pipe);

        if (display) {
            cv::imshow("Equalized Video", equalizedFrame);
            if (cv::waitKey(1) == 'q') break;
        }
    }

    cap.release();
    pclose(pipe);
    cv::destroyAllWindows();
}

void processFolder(const std::string& folder, const std::string& outputDir, bool display) {
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.path().extension() == ".avi" || entry.path().extension() == ".mp4") {
            processVideo(entry.path().string(), outputDir, display);
        }
    }
}

int main() {
    std::string path, outputDir;
    std::cout << "Enter input folder path: ";
    std::cin >> path;
    std::cout << "Enter output folder path: ";
    std::cin >> outputDir;
    std::cout << "Display video? (true/false): ";
    std::string displayInput;
    std::cin >> displayInput;
    bool display = (displayInput == "true");

    processFolder(path, outputDir, display);
    return 0;
}
