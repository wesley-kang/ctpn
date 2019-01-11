/*
	author: hengk
	date: 2019 01 09
	email:hengk@foxmail.com
	module : generate some ground true
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void generate_single_anchors(int f_w, int f_h, int feat_stride, float **anchors);
void cal_overlaps(float **a, int len1, float **b, int len2, float **ret);


typedef struct anchor_info{

	int type; // fore-ground 1 and back-ground 0  don't care -1


	float dx; // x's bias  ratio
	float dy; // y's bias  ratio
	float dw; // w's scale ratio
	float dh; // h's scale ratio

	float inside_weight_dx;
	float inside_weight_dy;
	float inside_weight_dw;
	float inside_weight_dh;

	float outside_weight_dx;
	float outside_weight_dy;
	float outside_weight_dw;
	float outside_weight_dh;
}anchor_info;

void transform(float ex_roi[4], float gt_roi[4], float  target[4])
{
	float ex_width = ex_roi[2] - ex_roi[0] + 1;
	float ex_height = ex_roi[3] - ex_roi[1] + 1;
	float ex_ctr_x = ex_roi[0] + 0.5*ex_width;
	float ex_ctr_y = ex_roi[1] + 0.5*ex_height;


	float gt_width = gt_roi[2] - gt_roi[0] + 1;
	float gt_height = gt_roi[3] - gt_roi[1] + 1;
	float gt_ctr_x = gt_roi[0] + 0.5*gt_width;
	float gt_ctr_y = gt_roi[1] + 0.5*gt_height;


	target[0] = (gt_ctr_x - ex_ctr_x) / ex_width;
	target[1] = (gt_ctr_y - ex_ctr_y) / ex_height;
	target[2] = log(gt_width / ex_width);
	target[3] = log(gt_height / ex_height);

}

float max_overlap(float **a, int len1,int len2,int axis,int *index)
{
	float max = 0.0;
	//search by cols
	if (axis == 1)
	{
		for (int i = 0; i < len2; i++)
		{

			if (a[len1][i]>max) {
				max = a[len1][i];
				*index = i;
			}
		}

	}
	//search by rows
	else
	{
		for (int i = 0; i < len1; i++)
		{
			if (a[i][len2]>max){
				max = a[i][len2];
				*index = i;
			}
		}
	}
	return max;
}

int check(float *coord, int len,int width, int height)
{
	for (int i = 0; i < len; i++)
	{
		// x coord
		if (i % 2 == 0)
		{
			if (coord[i] < 0 || coord[i] >= width)
				return 0;
		}
		//y coord
		else
		{
			if (coord[i] < 0 || coord[i] >= height)
				return 0;
		}
	}
	return 1;
}

void generate_labels_bboxes(float **gt_boxes,int gt_len,int fea_h,int fea_w,int height,int width,int stride,float **ret)
{

	int anchors_len = fea_h*fea_w * 10;

	float **anchors = (float**)malloc(sizeof(float*)*anchors_len);
	float **overlaps = (float**)malloc(sizeof(float*)*anchors_len);
	anchor_info **anchors_info = (anchor_info **)malloc(sizeof(anchor_info*)*anchors_len);


	for (int i = 0; i < anchors_len; i++)
	{
		anchors[i] = (float*)malloc(sizeof(float)* 4);
		overlaps[i] = (float*)malloc(sizeof(float)*gt_len);
		anchors_info[i] = (anchor_info*)malloc(sizeof(anchor_info));
	}

    //generate anchors of a picture
	generate_single_anchors(fea_w, fea_h, stride, anchors);
	//cal overlaps between anchors and ground truth
	cal_overlaps(anchors, anchors_len, gt_boxes, gt_len, overlaps);

	for (int i = 0; i < anchors_len; i++)
	{


		//if the anchor is out of broad
		if (check(anchors[i], 4,width,height)==0)
		{
			anchors_info[i]->type = -1;

			anchors_info[i]->inside_weight_dx = 0;
			anchors_info[i]->inside_weight_dy = 0;
			anchors_info[i]->inside_weight_dw = 0;
			anchors_info[i]->inside_weight_dh = 0;

			anchors_info[i]->outside_weight_dx = 0;
			anchors_info[i]->outside_weight_dy = 0;
			anchors_info[i]->outside_weight_dw = 0;
			anchors_info[i]->outside_weight_dh = 0;

			anchors_info[i]->dx = 0;
			anchors_info[i]->dy = 0;
			anchors_info[i]->dw = 0;
			anchors_info[i]->dh = 0;
			continue;

		}

		int index = 0;
		float iou = max_overlap(overlaps,i,gt_len,1,&index);

		if (iou >= 0.7)
		{
			anchors_info[i]->type = 1;
			anchors_info[i]->inside_weight_dx = 1;
			anchors_info[i]->inside_weight_dy = 1;
			anchors_info[i]->inside_weight_dw = 1;
			anchors_info[i]->inside_weight_dh = 1;

			anchors_info[i]->outside_weight_dx = 1;
			anchors_info[i]->outside_weight_dy = 1;
			anchors_info[i]->outside_weight_dw = 1;
			anchors_info[i]->outside_weight_dh = 1;

			float target[4];

			transform(anchors[i],gt_boxes[index],target);

			anchors_info[i]->dx = target[0];
			anchors_info[i]->dy = target[1];
			anchors_info[i]->dw = target[2];
			anchors_info[i]->dh = target[3];

		}
		else if (iou <= 0.3)
		{
			anchors_info[i]->type = 0;
			anchors_info[i]->inside_weight_dx = 0;
			anchors_info[i]->inside_weight_dy = 0;
			anchors_info[i]->inside_weight_dw = 0;
			anchors_info[i]->inside_weight_dh = 0;

			anchors_info[i]->outside_weight_dx = 0;
			anchors_info[i]->outside_weight_dy = 0;
			anchors_info[i]->outside_weight_dw = 0;
			anchors_info[i]->outside_weight_dh = 0;

			anchors_info[i]->dx = 0;
			anchors_info[i]->dy = 0;
			anchors_info[i]->dw = 0;
			anchors_info[i]->dh = 0;

		}
		else
		{
			anchors_info[i]->type = -1;
			anchors_info[i]->inside_weight_dx = 0;
			anchors_info[i]->inside_weight_dy = 0;
			anchors_info[i]->inside_weight_dw = 0;
			anchors_info[i]->inside_weight_dh = 0;

			anchors_info[i]->outside_weight_dx = 0;
			anchors_info[i]->outside_weight_dy = 0;
			anchors_info[i]->outside_weight_dw = 0;
			anchors_info[i]->outside_weight_dh = 0;

			anchors_info[i]->dx = 0;
			anchors_info[i]->dy = 0;
			anchors_info[i]->dw = 0;
			anchors_info[i]->dh = 0;
		}

	}

	for (int i = 0; i < gt_len; i++)
	{
		int index = 0;
		max_overlap(overlaps, anchors_len, i, 0, &index);
		anchors_info[index]->type = 1;
		anchors_info[index]->inside_weight_dx = 1;
		anchors_info[index]->inside_weight_dy = 1;
		anchors_info[index]->inside_weight_dw = 1;
		anchors_info[index]->inside_weight_dh = 1;

		anchors_info[index]->outside_weight_dx = 1;
		anchors_info[index]->outside_weight_dy = 1;
		anchors_info[index]->outside_weight_dw = 1;
		anchors_info[index]->outside_weight_dh = 1;

		float target[4];
		transform(anchors[index], gt_boxes[i], target);

		anchors_info[index]->dx = target[0];
		anchors_info[index]->dy = target[1];
		anchors_info[index]->dw = target[2];
		anchors_info[index]->dh = target[3];


	}
    int neg =0;
    int pos =0;
    for(int i=0;i<anchors_len;i++)
    {

        if(anchors_info[i]->type == 1)
            pos++;
        else if(anchors_info[i]->type ==0)
            neg++;
    }

    //make the pos and neg samples equal
    int res = neg - pos;
    srand(1);
    int m=0;
    while(res)
    {

        float prob = (rand()%11)/10.0;
        if(anchors_info[m]->type ==0 && prob>0.5)
        {
            anchors_info[m]->type = -1;
            res--;
        }
         m= (m+1)%anchors_len;
    }

    for(int i=0;i<anchors_len;i++)
    {
        ret[i][0] = anchors_info[i]->type;

        ret[i][1] = anchors_info[i]->dx;
        ret[i][2] = anchors_info[i]->dy;
        ret[i][3] = anchors_info[i]->dw;
        ret[i][4] = anchors_info[i]->dh;


        ret[i][5] = anchors_info[i]->inside_weight_dx;
        ret[i][6] = anchors_info[i]->inside_weight_dy;
        ret[i][7] = anchors_info[i]->inside_weight_dw;
        ret[i][8] = anchors_info[i]->inside_weight_dh;

        ret[i][9] =  anchors_info[i]->outside_weight_dx;
        ret[i][10] = anchors_info[i]->outside_weight_dy;
        ret[i][11] = anchors_info[i]->outside_weight_dw;
        ret[i][12] = anchors_info[i]->outside_weight_dh;



    }
//     neg =0;
//     pos =0;
//     for(int i=0;i<anchors_len;i++)
//    {
//
//        if(anchors_info[i]->type == 1)
//            pos++;
//        else if(anchors_info[i]->type ==0)
//            neg++;
//    }
//    printf("%d %d\n",neg,pos);
	//free memory
	for (int i = 0; i < anchors_len; i++)
	{
	    free(overlaps[i]);
		free(anchors[i]);
	    free(anchors_info[i]);
	}

	free(anchors);
	free(anchors_info);
	free(overlaps);
}
