#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


bool get_file_info(const string file_dir, const string suffix, vector<string> &sorted_names) {
    string path = file_dir + "/" + suffix;
    vector<String> fn;
    cv::glob(path, fn, false);
    size_t count = fn.size();
    if (count == 0) {
        cout << "file " << path << " not  exits" << endl;
        return -1;
    }
    string::size_type iPos = fn[0].find_last_of('/') + 1;
    for (int i = 0; i < count; ++i) {
        string filename = fn[i].substr(iPos, fn[i].length() - iPos);
        string name = filename.substr(0, filename.rfind("."));
        sorted_names.emplace_back(name);
    }
    sort(sorted_names.begin(), sorted_names.end());
    return false;
}

int main() 
{
    string data_dir = "/home/SENSETIME/sunxin/3_datasets/WHDLD/ImagesPNG/";

    string suffix = "*.png";

    string save_dir = "/home/SENSETIME/sunxin/3_datasets/WHDLD/Masks/";

    string vis_dir = "/home/SENSETIME/sunxin/3_datasets/WHDLD/MasksVis/";

    vector<string> names;
    get_file_info(data_dir, suffix, names);
    for (size_t i = 0; i < names.size(); i++) 
    {
        string scene = names[i];

        string dir_tmp = data_dir;
        string png_path = dir_tmp.append(scene + ".png");
        cout << png_path << endl;

        string dir_tmp2 = save_dir;
        string png_path2 = dir_tmp2.append(scene + ".png");

        string dir_tmp3 = vis_dir;
        string png_path3 = dir_tmp3.append(scene + ".png");

        Mat image = imread(png_path);
        cvtColor(image, image, CV_BGR2RGB);
        Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
        for(int i=0;i<image.rows;i++)  
        {  
            for(int j=0;j<image.cols;j++)  
            {  
                int r = image.at<Vec3b>(i,j)[0];
                int g = image.at<Vec3b>(i,j)[1];  
                int b = image.at<Vec3b>(i,j)[2];
                int f = 6;

                if ((r==192)&&(g==192)&&(b==0))
                    f = 0;
                else if ((r==0)&&(g==255)&&(b==0))
                    f = 1;
                else if ((r==128)&&(g==128)&&(b==128))
                    f = 2;
                else if ((r==255)&&(g==255)&&(b==0))
                    f = 3;
                else if ((r==255)&&(g==0)&&(b==0))
                    f = 4;
                else if ((r==0)&&(g==0)&&(b==255))
                    f = 5;
                else
                {
                    cout << "Error! Found undefined color!" << endl;
                    return -1;
                }
                
                mask.at<uchar>(i, j) = f;
            }
        }
        std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(0);    // 无压缩png.
        compression_params.push_back(cv::IMWRITE_PNG_STRATEGY);
        compression_params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);  
        imwrite(png_path2, mask, compression_params);

        Mat vis = Mat::zeros(image.rows, image.cols, CV_8UC3);
        cvtColor(vis, vis, CV_BGR2RGB);
        for(int i=0;i<image.rows;i++)  
        {  
            for(int j=0;j<image.cols;j++)  
            {  
                int f = mask.at<uchar>(i, j);

                if (f==0)
                {
                    vis.at<Vec3b>(i,j)[0] = 192;
                    vis.at<Vec3b>(i,j)[1] = 192;  
                    vis.at<Vec3b>(i,j)[2] = 0;
                }
                else if (f==1)
                {
                    vis.at<Vec3b>(i,j)[0] = 0;
                    vis.at<Vec3b>(i,j)[1] = 255;  
                    vis.at<Vec3b>(i,j)[2] = 0;
                }
                else if (f==2)
                {
                    vis.at<Vec3b>(i,j)[0] = 128;
                    vis.at<Vec3b>(i,j)[1] = 128;  
                    vis.at<Vec3b>(i,j)[2] = 128;
                }
                else if (f==3)
                {
                    vis.at<Vec3b>(i,j)[0] = 255;
                    vis.at<Vec3b>(i,j)[1] = 255;  
                    vis.at<Vec3b>(i,j)[2] = 0;
                }
                else if (f==4)
                {
                    vis.at<Vec3b>(i,j)[0] = 255;
                    vis.at<Vec3b>(i,j)[1] = 0;  
                    vis.at<Vec3b>(i,j)[2] = 0;
                }
                else if (f==5)
                {
                    vis.at<Vec3b>(i,j)[0] = 0;
                    vis.at<Vec3b>(i,j)[1] = 0;  
                    vis.at<Vec3b>(i,j)[2] = 255;
                }
                else
                {
                    cout << "Error! Found undefined label!" << endl;
                    return -1;
                }
            }
        }
        cvtColor(vis, vis, CV_RGB2BGR);
        imwrite(png_path3, vis);
    }
    return 0;
}

