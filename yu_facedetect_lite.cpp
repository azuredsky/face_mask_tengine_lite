#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <dirent.h>

#include "tengine_c_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

float show_threshold = 0.5;

struct Box
{
    Rect r;
    float score;
    float mask;
};

struct Anchor
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

void get_input_data(cv::Mat &img, float *input_data, int img_h, int img_w)
{
    int mean[3] = {104, 117, 123};
    unsigned char *src_ptr = (unsigned char*) (img.ptr(0));
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (float) (*src_ptr);
//                input_data[c * hw + h * img_w + w] = (float) (*src_ptr);
                src_ptr++;
            }
        }
    }
}

void get_image_names(std::string file_path,
                     std::vector<std::string> &file_names)
{
    DIR *dir;
    struct dirent *ptr;
    dir = opendir(file_path.c_str());
    while ( (ptr = readdir(dir)) != NULL)
    {
        string filename = string(ptr->d_name);
        if (filename == "." || filename == "..")
        {
            continue;
        }
        string path = file_path + string("/") + filename;
        file_names.push_back(path);
    }
    closedir(dir);
    sort(file_names.begin(), file_names.end());
}

void gen_anchors(vector<Anchor>&anchors, int img_w, int img_h)
{

    int feature_map_2th[2] = {img_w / 2 / 2, img_h / 2 / 2};
    vector<Point>feature_map;
    feature_map.push_back(Point(feature_map_2th[0] / 2,
                                feature_map_2th[1] / 2));
    feature_map.push_back(Point(feature_map_2th[0] / 4,
                                feature_map_2th[1] / 4));
    feature_map.push_back(Point(feature_map_2th[0] / 8,
                                feature_map_2th[1] / 8));
    feature_map.push_back(Point(feature_map_2th[0] / 16,
                                feature_map_2th[1] / 16));
    int steps[] = {8, 16, 32, 64};
    map<int, vector<int>>min_sizes;
    min_sizes[0] = {10, 16, 24};
    min_sizes[1] = {32, 48};
    min_sizes[2] = {64, 96};
    min_sizes[3] = {128, 192, 256};

    for (auto k = 0; k < feature_map.size(); k++)
    {
        for (auto j = 0; j < feature_map[k].y; j++)
        {
            for (auto i = 0; i < feature_map[k].x; i++)
            {
                for (auto m = 0; m < min_sizes[k].size(); m++)
                {
                    Anchor anchor;
                    float cx = (i + 0.5f) * steps[k] / img_w;
                    float cy = (j + 0.5f) * steps[k] / img_h;

                    float sw = min_sizes[k][m] * 1.f / img_w;
                    float sh = min_sizes[k][m] * 1.f / img_h;

                    // xmin
                    anchor.xmin = (cx - sw / 2.f);
                    // ymin
                    anchor.ymin = (cy - sh / 2.f);
                    // xmax
                    anchor.xmax = (cx + sw / 2.f);
                    // ymax
                    anchor.ymax = (cy + sh / 2.f);

                    anchors.push_back(anchor);
                }
            }
        }
    }

}

static void qsort_descent_inplace(std::vector<Box> &objects,
                                  int left,
                                  int right)
{
    int i = left;
    int j = right;
    float p = objects[ (left + right) / 2].score;

    while (i <= j)
    {
        while (objects[i].score > p)
            i++;

        while (objects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Box> &objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Box &a, const Box &b)
{
    cv::Rect_<float> inter = a.r & b.r;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Box> &objects,
                              std::vector<int> &picked,
                              float NMS_THRES)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].r.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Box &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Box &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > NMS_THRES)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int main(int argc, char *argv[])
{

    std::string tm_file = "facedetectcnn.tmfile";

    int repeat_count = 1;

    vector<string>file_names;
    size_t count = 0;

    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cv::Mat img;
    string video_name;

    init_tengine();
    cout<<get_tengine_version()<<endl;

    double s = (double)cv::getTickCount();

    vector<Anchor>anchors;
    gen_anchors(anchors, 320, 240);

    while (1)
    {

        double t = (double)cv::getTickCount();

        graph_t graph = create_graph(nullptr, "tengine", tm_file.c_str());
        if (graph == nullptr)
        {
            std::cout << "Create graph0 failed\n";
            std::cout << "errno: " << get_tengine_errno() << "\n";
            return -1;
        }

        /* get input tensor */
        cap >> img;
        if (img.empty())
        {
            std::cerr << "failed to read image file " << "\n";
            return -1;
        }
#if 1
        // resize to 320 x 240
        cv::Mat resize_img;
        int img_w = 320;
        int img_h = 240;
        cv::resize(img,
                   resize_img,
                   cv::Size(img_w, img_h),
                   0,
                   0,
                   cv::INTER_LINEAR);
        float *input_data = (float*)malloc(sizeof(float) * img_h * img_w * 3);

        get_input_data(resize_img, input_data, img_h, img_w);

#else
        // use origin image size
        int img_h = img.rows;
        int img_w = img.cols;
        float *input_data = (float*)malloc(sizeof(float) * img_h * img_w * 3);
        get_input_data(img, input_data, img_h, img_w);
#endif
        int node_idx = 0;
        int tensor_idx = 0;
        tensor_t input_tensor = get_graph_input_tensor(graph,
                                                       node_idx,
                                                       tensor_idx);
        int dims[] = {1, 3, img_h, img_w};
        set_tensor_shape(input_tensor, dims, 4);
        /* setup input buffer */
        if (set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
        {
            std::printf("Set buffer for tensor failed\n");
            return -1;
        }

        int rt = prerun_graph(graph);
        // time run_graph

        const char *repeat = std::getenv("REPEAT_COUNT");
        if (repeat)
            repeat_count = std::strtoul(repeat, NULL, 10);

        struct timeval t0, t1;
        float avg_time = 0.f;
        gettimeofday(&t0, NULL);
        for (int i = 0; i < repeat_count; i++)
            run_graph(graph, 1);
        gettimeofday(&t1, NULL);
        float mytime = (float) ( (t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        std::cout << "--------------------------------------\n";
        std::cout << "repeat " << repeat_count << " times, avg time per run is "
                  << avg_time / repeat_count << " ms\n";

        // post process
        char clsname[100] = "333";
        char regname[100] = "322";
        char ptsname[100] = "343";

        tensor_t clsBlob = get_graph_tensor(graph, clsname);    // * 2
        tensor_t regBlob = get_graph_tensor(graph, regname);    // * 14
        tensor_t maskBlob = get_graph_tensor(graph, ptsname);   // * 1

        int cls_dims[4];
        int reg_dims[4];
        int pts_dims[4];

        get_tensor_shape(clsBlob, cls_dims, 4);
        get_tensor_shape(regBlob, reg_dims, 4);
        get_tensor_shape(maskBlob, pts_dims, 4);

        const float *clsData = (float*)get_tensor_buffer(clsBlob);
        const float *regData = (float*)get_tensor_buffer(regBlob);
        const float *maskData = (float*)get_tensor_buffer(maskBlob);

        vector<Box>boxes;
        float prior_variance[2] = {0.1f, 0.2f};

        int anchor_num = anchors.size();

        for (int i = 0; i < anchor_num; i++)
        {
            Box box;
            int face_idx = i * 2 + 1;

            if (clsData[face_idx] > 0.8)
            {
                float fBox_x1 = anchors[i].xmin;
                float fBox_y1 = anchors[i].ymin;
                float fBox_x2 = anchors[i].xmax;
                float fBox_y2 = anchors[i].ymax;

                float locx1 = regData[i * 14];
                float locy1 = regData[i * 14 + 1];
                float locx2 = regData[i * 14 + 2];
                float locy2 = regData[i * 14 + 3];

                float prior_width = fBox_x2 - fBox_x1;
                float prior_height = fBox_y2 - fBox_y1;
                float prior_center_x = (fBox_x1 + fBox_x2) / 2;
                float prior_center_y = (fBox_y1 + fBox_y2) / 2;

                float box_centerx = prior_variance[0] * locx1 * prior_width + prior_center_x;
                float box_centery = prior_variance[1] * locy1 * prior_height + prior_center_y;
                float box_width = expf(prior_variance[2] * locx2) * prior_width;
                float box_height = expf(prior_variance[3] * locy2) * prior_height;

                fBox_x1 = box_centerx - box_width / 2.f;
                fBox_y1 = box_centery - box_height / 2.f;
                fBox_x2 = box_centerx + box_width / 2.f;
                fBox_y2 = box_centery + box_height / 2.f;

                fBox_x1 = MAX(0, fBox_x1);
                fBox_y1 = MAX(0, fBox_y1);
                fBox_x2 = MIN(1.f, fBox_x2);
                fBox_y2 = MIN(1.f, fBox_y2);

                box.r.x = fBox_x1 * img_w;
                box.r.y = fBox_y1 * img_h;
                box.r.width = fBox_x2 * img_w - fBox_x1 * img_w;
                box.r.height = fBox_y2 * img_h - fBox_y1 * img_h;
                box.score = clsData[face_idx];
                box.mask = 1.0 / (1.0 + expf(-maskData[i]));

                boxes.push_back(box);
            }
        }

        qsort_descent_inplace(boxes);
        std::vector<int> picked;
        nms_sorted_bboxes(boxes, picked, 0.1);

        int face_count = picked.size();

        cv::Mat result = resize_img.clone();

        printf("detect faces : %d, nms = %d\n", boxes.size(), face_count);

        for (int i = 0; i < (int)picked.size(); i++)
        {
            Box box = boxes[picked[i]];
            string text = box.mask > 0.5 ? "mask" : "nomask";
            cv::rectangle(result, box.r, cv::Scalar(255, 255, 0), 1);
            cv::putText(result, text, box.r.tl(),1,1.0, cv::Scalar(255, 255, 0));
        }

        release_graph_tensor(input_tensor);
        postrun_graph(graph);
        destroy_graph(graph);
        free(input_data);

        cv::waitKey(1);

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        double fps = 1.0 / t;

        cv::putText(result,
                    "FPS: " + to_string(fps),
                    cv::Point(5, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255));
        cv::imshow("result", result);

        cout << count << " Time consumed: " << 1000*t << "ms" << "   FPS: " << fps
             << endl;

        count++;
    }

    s = ((double)cv::getTickCount() - s) / cv::getTickFrequency();

    cout << "Average FPS: " << (count + 1) / s << endl;

    release_tengine();

    return 0;
}
