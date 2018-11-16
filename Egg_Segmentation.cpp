#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>
#include <functional>
#include <queue>
#include <dirent.h>

using namespace cv;
using namespace std;

void display_image(Mat img)
{
	//	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", img);
	waitKey(0);
	cv::destroyWindow("Display window");
}
void fill_large_contour(Mat &im,double size)
{
	 if (im.channels() != 1 || im.type() != CV_8U)
	 {
		 return;
	 }
	 // Find all contours
	 std::vector<std::vector<cv::Point> > contours;
	 cv::findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	 for (int i = 0; i <(int) contours.size(); i++)
	 {
        double area = cv::contourArea(contours[i]);
	    if (area >size)
	    {
	    	cv::drawContours(im, contours, i, CV_RGB(0, 0, 0), 10);
	    }

	 }

}
Mat Sobel_edge_detection(Mat img_rgb)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat img_gray;
	if (img_rgb.channels()==3)
	{
		cvtColor(img_rgb, img_gray,CV_RGB2GRAY);
	}
	else
	{
		img_gray=img_rgb;
	}
	Mat grad_x, grad_y,img_Sobel;
	Mat abs_grad_x, abs_grad_y;
    Sobel( img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
    Sobel( img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img_Sobel );
	//display_image(img_Sobel);
	return (img_Sobel);
}
Mat Laplacian_edge_detection(Mat img_rgb)
{
	Mat tmp;
	Mat img_gray;
	if (img_rgb.channels()==3)
	{
		cvtColor(img_rgb, img_gray,CV_RGB2GRAY);
	}
	else
	{
		img_gray=img_rgb;
	}
	GaussianBlur( img_gray, img_gray, Size(3,3), 0, 0, BORDER_DEFAULT );
	Laplacian( img_gray, tmp, CV_16S, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( tmp, tmp );
	return (tmp);
}
void bwareaopen(cv::Mat& im, double size)
{
    // Only accept CV_8UC1
    if (im.channels() != 1 || im.type() != CV_8U)
    {
		cout<<"abc"<<endl;
    	return;
    }
  //  cv::bitwise_not(im, im);
    // Find all contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i <(int) contours.size(); i++)
    {
        // Calculate contour area
        double area = cv::contourArea(contours[i]);
		//cout<<"area= "<<area<<endl;
        // Remove small objects by drawing the contour with black color
       if (area >= 0 && area <= size)
       {
        	cv::drawContours(im, contours, i, CV_RGB(0, 0, 0), -1);
        }
    }
  //  cv::bitwise_not(im, im);
}

vector<Mat> egg_extraction(Mat img_rgb,string line)
{
	vector<Mat> subregions;
	//------chuyen anh mau ve den trang----------------
	Mat img_gray;
	if (img_rgb.channels()==3)
	{
		cvtColor(img_rgb, img_gray, cv::COLOR_RGB2GRAY);
	}
	else
	{
		img_gray=img_rgb;
	}
	//-----------------------------------------------
	//display_image(img_gray);

	//-----------Edge detection------------------------
	Mat img_canny;
	int thresh =30;
	int max_thresh = 255;
	Canny(img_gray, img_canny, thresh, thresh*3, 3 );
	Mat img_Sobel,img_Sobel_bin,img_Laplacian,img_Laplacian_bin;
	img_Sobel=Sobel_edge_detection(img_gray);
	display_image(img_canny);
	img_Laplacian=Laplacian_edge_detection(img_gray);
	threshold( img_Laplacian, img_Laplacian_bin,20,255,THRESH_BINARY );
	display_image(img_Laplacian_bin);
	threshold( img_Sobel, img_Sobel_bin, 20,255,THRESH_BINARY );
	display_image(img_Sobel_bin);
	Mat img_bin;
	//------------------------------------------------------

	//------------Ket hop 3 bo edge detection---------------
	bitwise_or(img_canny,img_Sobel_bin,img_bin);
	bitwise_or(img_bin,img_Laplacian_bin,img_bin);
	//bitwise_or(img_Sobel_bin,img_Laplacian_bin,img_bin);
	//display_image(img_bin);
	//----------------------------------------------------
	//display_image(img_Sobel_bin);
	vector<vector<Point> > contours;
	vector<vector<Point> > contours_ori;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	int morph_elem = 0;
	int morph_size = 0;
	int morph_operator = 0;
	// bwareaopen(img_bin,6000);
	// display_image(img_bin);
	for (int i=0;i<1;i++)
	{
		Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 1, 1), Point( -1, -1 ) );
		morphologyEx( img_bin, img_bin, MORPH_OPEN, element );
	}
	//display_image(img_bin);
	for (int i=0;i<1;i++)
	{
		bwareaopen(img_bin,100); // loai bo nhung objec co size nho hon 100
		fill_large_contour(img_bin,500000); // loai bo nhung object co size lon hon 500000
	}
	//display_image(img_bin);
	//line_connect(img_bin,10);
	//display_image(img_bin);
	//imwrite("img_bin.jpg",img_bin);
	for (int i=0;i<2;i++)
	{
		Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 7, 7 ), Point( -1, -1 ) );
		morphologyEx( img_bin, img_bin, MORPH_CLOSE, element );
	}
	display_image(img_bin);
	
	//---------------Xu li voi contour---------------------------
	findContours( img_bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
	contours_ori=contours;
	Mat drawing = Mat::zeros( img_canny.size(), CV_8UC3 );
	vector<vector<Point> >hull( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{
		convexHull( Mat(contours[i]), hull[i], false );
	}
	vector<vector<Point> > contours_approx( contours.size() );
	for( int i = 0; i < hull.size(); i++ )
	{
		approxPolyDP(hull[i], contours_approx[i],arcLength(Mat(contours[i]),true)*0.001, true);
	}
	//------------------------------------------------------------

	for( int i = 0; i< contours.size(); i++ )
	{
		//if ((contourArea(contours[i])>1000)&&(contourArea(contours[i])<50000))
		if ((contourArea(contours[i])>6000)&&(contourArea(contours[i])<500000)&&(contours_approx[i].size()>10))
		{
			Mat mask = Mat::zeros(img_rgb.size(), CV_8UC1);
			//----------------------Thu nho size of hull contour hear--------------------
			Point center;
			center.x=0;center.y=0;
			for (int j=0;j<hull[i].size();j++)
			{
				center.x+=hull[i][j].x;
				center.y+=hull[i][j].y;
			}
			center.x=(int) (center.x/hull[i].size());
			center.y=(int) (center.y/hull[i].size());
			//cout<<center<<endl;
			for (int j=0;j<hull[i].size();j++)
			{
				hull[i][j].x=hull[i][j].x+(int) (0.05*(center.x-hull[i][j].x));
				hull[i][j].y=hull[i][j].y+(int) (0.05*(center.y-hull[i][j].y));
			}
			//---------------------------------------------------------------------
			drawContours(mask, hull, i, Scalar(255), CV_FILLED);
			Mat contourRegion;
			Mat imageROI;
			Rect roi = boundingRect(contours_ori[i]);
			float ratio=(float)roi.height/roi.width;
			if ((ratio>0.5)&&(ratio<2))
			{
				img_rgb.copyTo(imageROI, mask); // 'image' is the image you used to compute the contours.
				//display_image(imageROI);
				contourRegion = imageROI(roi);
//				cout<<roi<<endl;
//				cout<<contourRegion.size()<<endl;
				//display_image(contourRegion);
				putText(img_rgb, to_string(subregions.size()), center, FONT_HERSHEY_DUPLEX, 5, Scalar(0,50,243), 5);
				display_image(contourRegion);
				cout<<"countour size is " <<contourArea(contours[i])<<endl;
				subregions.push_back(contourRegion);
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours( drawing, hull, i, color, 2, 8, hierarchy, 0, Point() );
			}
		//	drawContours( drawing, contours_approx , i, color, 2, 8, hierarchy, 0, Point() );
		//	display_image(contourRegion);

		//			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );\
		//			cout<<contours_approx[i].size()<<endl;
		//			drawContours( drawing, contours_approx, i, color, 2, 8, hierarchy, 0, Point() );
		}

	}
	char filename[200];
	sprintf(filename,"%s_index.jpg",line.c_str());
	imwrite(filename,img_rgb);
	//display_image(img_rgb);
	//display_image(drawing);
	return(subregions);
}

Mat img_transparent(Mat img_rgb)
{
	Mat tmp;
	Mat img_gray,img_bin;
	cout<<"type of image is "<<img_rgb.type()<<endl;
	if (img_rgb.channels()==3)
	{
		cvtColor(img_rgb, img_gray, cv::COLOR_RGB2GRAY);
	}
	else
	{
		img_gray=img_rgb;
	}
	threshold(img_gray, img_bin, 10, 255, CV_THRESH_BINARY);
	Mat mask = Mat::zeros(img_rgb.size(), CV_8UC1);
	Mat img_out(img_rgb.rows,img_rgb.cols,CV_8UC4);
	vector<vector<Point> > contours;
	vector<vector<Point> > contours_ori;
	vector<Vec4i> hierarchy;
	findContours( img_bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );
	//display_image(img_bin);
	RNG rng(12345);
	//cout<<"number of contours is "<<contours.size()<<endl;
	int max_index=0;
	float max_size=0;
	for (int i=0;i<contours.size();i++)
	{
		if (contourArea(contours[i])>max_size)
		{
			max_size=contourArea(contours[i]);
			max_index=i;
		}
	}
	//cout<<"max_index= "<<max_index;
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//drawContours( img_rgb, contours, max_index, color, 2, 8, hierarchy, 0, Point() );
	drawContours(mask, contours, max_index, Scalar(255), CV_FILLED);
	Mat bgra[4];
	Mat jpg[3];
	split(img_rgb,jpg);
	bgra[0]=jpg[0];
	bgra[1]=jpg[1];
	bgra[2]=jpg[2];
	bgra[3]=mask;
	merge(bgra,4,img_out);

	
	// for (int i=0;i<img_rgb.size().width;i++)
	// {
	// 	for (int j=0;j<img_rgb.size().height;j++)
	// 	{
	// 		Vec4b png;
	// 		Vec3b rgb=img_rgb.at<Vec3b>(j,i);
	// 		png[0]=rgb[0];
	// 		png[1]=rgb[1];
	// 		png[2]=rgb[2];
	// 		if (mask.at<uchar>(j,i)>0)
	// 		{
	// 			png[3]=1;
				
	// 		}
	// 		else
	// 		{
	// 			png[3]=0;
	// 		}
	// 		img_out.at<Vec4b>(j,i)=png;
	// 		//tmp[0]
	// 	}
	// }
	

	return (img_out);
}
int main()
{

	//--------------------Doc va hien thi Image-----------------------
	// Mat img;
	// img=imread("/home/tonlh/Desktop/Code/Fugo_Eggs/Photos_eggs_Noneyama/photos_2005/1_lanbuc_5-6_nido1_toshokan/1_lanbuc_5-6_nido1_(1).jpg",1);
	// display_image(img);
	//------------------------------------------------------------------
	//-------Egg_transparent-------------------------------------------
	ifstream myfile ("/home/tonlh/Desktop/Code/Fugo_Eggs/results_2004.txt");
	char filename[200];
	string line;
	 if (myfile)  // same as: if (myfile.good())
	 {
		int count=0;
	    while (getline( myfile, line ))  // same as: while (getline( myfile, line ).good())
	    {
	    	count++;
	    	if ((count>0)&&(count<207))
	    	{
	    		vector<Mat> tmp;
	    		Mat img_rgb=imread(line,1);
				line.erase (line.end()-4, line.end());
				sprintf(filename,"%s.png",line.c_str());
				cout<<"line is: "<<filename<<endl;
				Mat img_out=img_transparent(img_rgb);
				vector<int> compression_params;
    			compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    			compression_params.push_back(0);
				//display_image(img_out);
				imwrite(filename, img_out,compression_params);
				//display_image(img_rgb);
	    	}
	    }
	    myfile.close();
	 }




	// ifstream myfile ("/home/tonlh/Desktop/Code/Fugo_Eggs/filename2005.txt");
	// string line;
	//  if (myfile)  // same as: if (myfile.good())
	//  {
	// 	int count=0;
	//     while (getline( myfile, line ))  // same as: while (getline( myfile, line ).good())
	//     {
	//     	count++;
	//     	if ((count>0)&&(count<2))
	//     	{
	//     		vector<Mat> tmp;
	//     		Mat img_rgb=imread(line,1);
	// 			display_image(img_rgb);
	//     		tmp=egg_extraction(img_rgb,line);
	//     		char filename[200];
	//     		cout<<tmp.size()<<endl;
	//     		for (int i=0;i<tmp.size();i++)
	//     		{
	//     			sprintf(filename,"%s_%d.jpg",line.c_str(),i);
	//     			cout<<filename<<endl;
	//     			imwrite(filename,tmp[i]);
	//     		}
	//     	}
	//     }
	//     myfile.close();
	//  }

	return (0);
}
