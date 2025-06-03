/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

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
    cout << "Extracting   features..." << endl;

    if (use_nn) {
        Ort::SuperPoint osh("/dbow3/super_point.onnx", 1);

        for(size_t i = 0; i < path_to_images.size(); ++i)
        {
            cout << "reading image: "<< path_to_images[i] << endl;
            cv::Mat image = cv::imread(path_to_images[i], 0);
            if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);

            KeyPointAndDesc results = osh.inference(osh, image);
            std::vector<cv::KeyPoint> keypoints = results.first;
            // cv::Mat descriptors;
            // cv::normalize(results.second, descriptors, 1.0, 0.0, cv::NORM_L2);

             for (int i=0; i < results.second.rows; i++) {
                cv::normalize(results.second.row(i), results.second.row(i), 1.0, 0.0, cv::NORM_L2);
                // Вычисляем L2-норму дескриптора
                double norm = cv::norm(results.second.row(i), cv::NORM_L2);
                // Проверяем, близка ли норма к 1 (с учетом погрешности)
                bool is_normalized = std::abs(norm - 1.0) < 1e-5;
                std::cout << "Normalized descriptor check: " << is_normalized << std::endl; 
            }
            features.push_back(results.second);

            // features.push_back(descriptors);
            cout << "size: " << path_to_images.size() << "counter: " << counter << endl;
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
            cout << "size: " << path_to_images.size() << "counter: " << counter << endl;
            counter = counter + 1;
        }
    }

    return features;
}

ScoringType getScoringTypeFromStr(std::string scoringType) {
    if (scoringType == "L1_NORM") return ScoringType::L1_NORM;
    else if (scoringType == "L2_NORM") return ScoringType::L2_NORM;
    else  cerr << "Unknown scoring_type" << endl;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features, ScoringType score, std::string out_path)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "getDescritorSize: " << voc.getDescritorSize() << endl;
    cout << "getDescritorType: " << voc.getDescritorType() << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(size_t i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for(size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save(out_path);
    cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features, std::string voc_path)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc(voc_path);

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}

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
        if (cml["-h"] || argc<=3){
            cerr<<"Usage:  descriptor_name out_features images_dir scoring_type out_voc.yml[.gz] ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor = argv[1];
        auto out_features = argv[2];
        auto images = DirReader::read(argv[3]);
        auto scoring_type = getScoringTypeFromStr(argv[4]);
        auto out_voc = argv[5];
        vector<cv::Mat> features = loadFeatures(images, descriptor);

        //save features to file
        saveToFile(argv[2],features);

        testVocCreation(features, scoring_type, out_voc);
        testDatabase(features, out_voc);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
