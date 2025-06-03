
//First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

#include "ort-superpoint/SuperPoint.hpp"
#include "ort-superpoint/Utility.hpp"
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "dir_reader.h"

using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

using namespace DBoW3;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    bool use_nn = false;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
    #ifdef OPENCV_VERSION_3
        else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
    #endif
    #ifdef USE_CONTRIB
        else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
    #endif

    else if (descriptor == "superpoint") use_nn = true;

    else throw std::runtime_error("Invalid descriptor");

    int counter = 1;
    vector<cv::Mat> features;
    cout << "Extracting features..." << endl;
    auto images_num = path_to_images.size();
    auto log_freq = 20;

    if (use_nn) {
        Ort::SuperPoint osh("/fbow/super_point.onnx", 0);

        for(size_t i = 0; i < path_to_images.size(); ++i)
        {
            cv::Mat image = cv::imread(path_to_images[i], 0);
            if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);

            KeyPointAndDesc results = osh.inference(osh, image);
            std::vector<cv::KeyPoint> keypoints = results.first;
            cv::Mat descriptors;
            cv::normalize(results.second, descriptors, 1.0, 0.0, cv::NORM_L2);

            features.push_back(descriptors);
            if (counter % log_freq == 0)
                cout << "images_num: " << images_num << "counter: " << counter << endl;
            counter = counter + 1;
        }

    } else {
        assert(!descriptor.empty());

        for(size_t i = 0; i < path_to_images.size(); ++i)
        {
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            cout<<"reading image: "<<path_to_images[i]<<endl;
            cv::Mat image = cv::imread(path_to_images[i], 0);
            if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
            cout<<"extracting features"<<endl;

            fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);

            features.push_back(descriptors);
            if (counter % log_freq == 0)
                cout << "images_num: " << images_num << "counter: " << counter << endl;
            counter = counter + 1;
        }
    }

    return features;
}

// ----------------------------------------------------------------------------
void saveToFile(string filename,const vector<cv::Mat> &features){

    //test it is not created
    std::ifstream ifile(filename);
    if (ifile.is_open()){cerr<<"ERROR::: Output File "<<filename<<" already exists!!!!!"<<endl;exit(0);}
    std::ofstream ofile(filename);
    if (!ofile.is_open()){cerr<<"could not open output file"<<endl;exit(0);}
    uint32_t size=features.size();
    ofile.write((char*)&size,sizeof(size));
    for(auto &f:features){
        if( !f.isContinuous()){
            cerr<<"Matrices should be continuous"<<endl;exit(0);
        }
        uint32_t aux=f.cols; ofile.write( (char*)&aux,sizeof(aux));
          aux=f.rows; ofile.write( (char*)&aux,sizeof(aux));
          aux=f.type(); ofile.write( (char*)&aux,sizeof(aux));
        ofile.write( (char*)f.ptr<uchar>(0),f.total()*f.elemSize());
    }
}

// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc==1){
            cerr<<"Usage:  descriptor_name output images_dir ... \n\t descriptors:superpoint,brisk,surf,orb(default),akaze(only if using opencv 3)"<<endl;
            return -1;
        }

        string descriptor=argv[1];
        string output=argv[2];

        auto images = DirReader::read(argv[3]);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);

        //save features to file
        saveToFile(argv[2],features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
