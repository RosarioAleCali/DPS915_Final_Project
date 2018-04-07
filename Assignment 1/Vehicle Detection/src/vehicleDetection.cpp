/*
    Author: Rosario A. Cali
    
    Description:
    This code in an adaptation from two dlib examples that can be found
    here http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html and
    here http://dlib.net/dnn_mmod_find_cars_ex.cpp.html
    The scope of this project is to Decode a video,
    track possible vehicles in the frames, and encode it again

    Compile using the following command:
    g++ -std=c++11 -O3 -I <path>/dlib-19.9/  <path>/dlib-19.9/dlib/all/source.cpp -lpthread -lX11 vehicleDetection.cpp -lpng -DDLIB_PNG_SUPPORT -o vehicleDetector

    Example:
    g++ -std=c++11 -O3 -I /home/rosarioalecali/dlib-19.9/  /home/rosarioalecali/dlib-19.9/dlib/all/source.cpp -lpthread -lX11 vehicleDetection.cpp -lpng -DDLIB_PNG_SUPPORT -o vehicleDetector

    Note, the following programs and libraries need to be installed
    for a correct usage:
    - FFmpeg
    - Dlib
    - libpng-dev
*/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

using namespace dlib;

// The front and rear view vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<55, SUBNET>>>;
using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

net_type net;
shape_predictor sp;
matrix<rgb_pixel> img;
std::vector<file> files;
image_window win;

void detect_vehicles() {
    for (unsigned int i = 0; i < files.size(); i++) {
        // Load one image at the time and display it
        load_image(img, files[i]);
		win.set_image(img);

        // Run the detector on the image and show the output
        for (auto&& d : net(img)) {
            auto fd = sp(img, d);
			rectangle rect;

			for (unsigned long j = 0; j < fd.num_parts(); ++j)
				rect += fd.part(j);

			if (d.label == "rear")
				win.add_overlay(rect, rgb_pixel(255, 0, 0), d.label);
			else
				win.add_overlay(rect, rgb_pixel(255, 255, 0), d.label);
        }

        // Clear the overlay
        dlib::sleep(1000);
        win.clear_overlay();
    }
}

int main(int argc, char *argv[]) try {
    if (argc != 2) {
        std::cerr << "Incorrect number of arguments!" << std::endl;
        std::cerr << "Correct usage:" << std::endl;
        std::cerr << "program-name <path-to-video>" << std::endl;
        return -1;
    }

    // Variables Declaration
    int returnCode;
    char command[126];

    // Initializing model
    std::cout << "Initializing model..." << std::endl;
    deserialize("mmod_front_and_rear_end_vehicle_detector.dat") >> net >> sp;
    std::cout << "Initialization done!" << std::endl;
    
    // Create tmp foldars to save images
    returnCode = system("mkdir tmp_frame_in");
    if (returnCode != 0) {
        std::cerr << "Error when creating tmp_frame_in directory." << std::endl;
        return -3;
    }

    // Create command to convert video frames to images
    memset(command, '\0', 126);
    strcat(command, "ffmpeg -i \"");
    strcat(command, argv[1]);
    strcat(command, "\" -vf fps=25 tmp_frame_in/thumb%06d.png -hide_banner");

    std::cout << command << std::endl;

    // Use FFmpeg to extract frames and save them to pictures
    std::cout << "Converting Video to Images (25fps)..." << std::endl;
    returnCode = system(command);
    if (returnCode != 0) {
        std::cerr << "Error occured! Make sure FFmpeg is installed by issuing the following command:" << std::endl;
        std::cerr << "sudo apt-get install ffmpeg" << std::endl;
        std::cerr << "If the program is installed, make sure the path to the video is correct." << std::endl;
        return -2;
    }
    std::cout << "Conversion done!" << std::endl << std::endl;

    // Get the list of video frames.  
  	files = get_files_in_directory_tree("tmp_frame_in", match_ending(".png"));
    std::sort(files.begin(), files.end());

    // Detect Vehicles
    detect_vehicles();

    // Removing tmp folder and all of its contents
    returnCode = system("rm -r tmp_frame_in");
    if (returnCode != 0) {
        std::cerr << "Error when deleting tmp_frame_in directory." << std::endl;
        return -3;
    }

    return 0;
}
catch (image_load_error& e)
{
	std::cerr << e.what() << std::endl;
	std::cerr << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << std::endl;
}
catch (serialization_error& e)
{
	std::cerr << e.what() << std::endl;
	std::cerr << "The correct model file can be obtained from: http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2" << std::endl;
}
catch (std::exception& e)
{
	std::cerr << e.what() << std::endl;
}