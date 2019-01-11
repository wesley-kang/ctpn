/*
	author: hengk
	date: 2018 12 04
	email:hengk@foxmail.com
	module : some utils used for handling anchors
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct Box{
	int no;
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int flag;
}box;

//bubble sort by hengk
void sort(box boxes[], int len)
{
	for (int i = 0; i < len - 1; i++)
	{
		for (int j = 0; j < len - i - 1; j++)
		{
			box tmp;
			if (boxes[j].score < boxes[j + 1].score)
			{
				tmp = boxes[j];
				boxes[j] = boxes[j + 1];
				boxes[j + 1] = tmp;
			}
		}
	}
}
float min1(float a, float b)
{
	return a > b ? b : a;
}
float max1(float a, float b)
{
	return a > b ? a : b;
}
//cal the iou between two boxes
float iou(box a1, box a2)
{
	float x1 = a1.x1;
	float x2 = a1.y1;
	float x3 = a1.x2;
	float x4 = a1.y2;

	float y1 = a2.x1;
	float y2 = a2.y1;
	float y3 = a2.x2;
	float y4 = a2.y2;

	float a1_area = (x3 - x1 + 1)*(x4 - x2 + 1);
	float a2_area = (y3 - y1 + 1)*(y4 - y2 + 1);

	float overlap_x_min = max1(x1, y1);
	float overlap_y_min = max1(x2, y2);
	float overlap_x_max = min1(x3, y3);
	float overlap_y_max = min1(x4, y4);

	float overlap_area = 0;
	if (overlap_x_max - overlap_x_min > 0 && overlap_y_max - overlap_y_min > 0)
	{

		overlap_area = (overlap_x_max - overlap_x_min + 1) * (overlap_y_max - overlap_y_min + 1);
		return overlap_area / (a1_area + a2_area - overlap_area);
	}
	return overlap_area;
}
/*
    dets: [][5]
    inds: []
*/
void nms(float **dets, int len, float thresh, int *inds, int *inds_len)
{

	box *boxes = (box*)malloc(sizeof(box)*len);
	//	printf("%d %f %d \n",len,thresh,*inds_len);
	for (int i = 0; i < len; i++)
	{
		//		printf("%f,%f,%f,%f,%f\n",dets[i][0],dets[i][1],dets[i][2],dets[i][3],dets[i][4]);
		boxes[i].no = i;
		boxes[i].x1 = dets[i][0];
		boxes[i].y1 = dets[i][1];
		boxes[i].x2 = dets[i][2];
		boxes[i].y2 = dets[i][3];
		boxes[i].score = dets[i][4];
		boxes[i].flag = 1;
	}
	//order by score
	sort(boxes, len);

	int total_size = len;
	int inds_index = 0;
	while (total_size > 0)
	{
		//select the highest score box from the boxes
		int index = 0;
		while (boxes[index].flag == 0) {
			index++;
		}
		inds[inds_index++] = boxes[index].no;
		//cal the iou of the  highest box with other's boxes and  throw away  the box that  (iou > thresh)
		int start = index;
		int end = len - 1;
		while (start <= end)
		{
			if(boxes[start].flag==0)
			{
				start++;
				continue;
			}
			float overlap = iou(boxes[index], boxes[start]);
			if (overlap > thresh)
			{
				boxes[start].flag = 0;
				total_size--;
			}
			start++;
		}
	}
	*inds_len = inds_index;
	free(boxes);
}
/*
    a:[][4]
    b:[][4]
    ret:[len1][len2]
*/
void cal_overlaps(float **a, int len1, float **b, int len2, float **ret)
{
	int i = 0,j = 0;
	box *boxes1 = (box*)malloc(sizeof(box)*len1);
	for ( i = 0; i < len1; i++)
	{
		boxes1[i].x1 = a[i][0];
		boxes1[i].y1 = a[i][1];
		boxes1[i].x2 = a[i][2];
		boxes1[i].y2 = a[i][3];
	}
	box *boxes2 = (box*)malloc(sizeof(box)*len2);
	for ( i = 0; i < len2; i++)
	{
		boxes2[i].x1 = b[i][0];
		boxes2[i].y1 = b[i][1];
		boxes2[i].x2 = b[i][2];
		boxes2[i].y2 = b[i][3];
	}

	for (int i = 0; i < len1; i++)
	{
		for (int j = 0; j < len2; j++)
		{
			ret[i][j] = iou(boxes1[i],boxes2[j]);
		}
	}

	free(boxes1);
	free(boxes2);
}
//generate the targets (the  params that need learn from network)
void bbox_transform(float ex_rois[][4], float gt_rois[][4], float targets[][4],int len)
{
	for (int i = 0; i < len; i++)
	{
		float ex_width = ex_rois[i][2] - ex_rois[i][0] + 1;
		float ex_height = ex_rois[i][3] - ex_rois[i][1] + 1;
		float ex_ctr_x = ex_rois[i][0] + 0.5*ex_width;
		float ex_ctr_y = ex_rois[i][1] + 0.5*ex_height;


		float gt_width = gt_rois[i][2] - gt_rois[i][0] + 1;
		float gt_height = gt_rois[i][3] - gt_rois[i][1] + 1;
		float gt_ctr_x = gt_rois[i][0] + 0.5*gt_width;
		float gt_ctr_y = gt_rois[i][1] + 0.5*gt_height;


		targets[i][0] = (gt_ctr_x - ex_ctr_x) / ex_width;
		targets[i][1] = (gt_ctr_y - ex_ctr_y) / ex_height;
		targets[i][2] = log(gt_width / ex_width);
		targets[i][3] = log(gt_height / ex_height);

	}
}
/*
    boxes:[][4]
    deltas:[][4]
    proposals:[][4]
    function : use the learning params to generate the proposals
*/
void bbox_transform_inv(float **boxes, float **deltas,float **proposals,int len)
{
	for (int i = 0; i < len; i++)
	{
		float width = boxes[i][2] - boxes[i][0] + 1;
		float height = boxes[i][3] - boxes[i][1] + 1;
		float ctr_x = boxes[i][0] + 0.5*width;
		float ctr_y = boxes[i][1] + 0.5*height;

		float dx = deltas[i][0];
		float dy = deltas[i][1];
		float dw = deltas[i][2];
		float dh = deltas[i][3];


		float pred_ctr_x = dx*width + ctr_x;
		float pred_ctr_y = dy*height + ctr_y;
		float pred_width = exp(dw)*width;
		float pred_height = exp(dh)*height;

		proposals[i][0] = pred_ctr_x - 0.5 * pred_width;
		proposals[i][1] = pred_ctr_y - 0.5 * pred_height;
		proposals[i][2] = pred_ctr_x + 0.5 * pred_width;
		proposals[i][3] = pred_ctr_y + 0.5 * pred_height;
	}

}

void generate_basic_anchors(float widths[],float heights[],int len,
				int basesize,float **anchors)
{
	float base_anchor[] = { 0, 0, basesize - 1, basesize - 1 };
	float x_ctr = (base_anchor[0] + base_anchor[2]) / 2;
	float y_ctr = (base_anchor[1] + base_anchor[3]) / 2;

	for (int i = 0; i < len; i++)
	{
		anchors[i][0] = x_ctr - widths[i]/2;
		anchors[i][1] = y_ctr - heights[i]/2;
		anchors[i][2] = x_ctr + widths[i]/2;
		anchors[i][3] = y_ctr + heights[i]/2;
	}
}
/*
    anchors:[][4]
*/
void generate_single_anchors(int f_w, int f_h, int feat_stride, float **anchors)
{
	float anchor_widths[] = { 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 };
	float anchor_heights[] = {11, 16, 23, 33, 48, 68, 97, 139,198,283};
	int anchor_num = sizeof(anchor_widths) / sizeof(float);
	int basesize = 16;


	float **base_anchor = (float**)malloc(sizeof(float*)*anchor_num);
	for (int i = 0; i < anchor_num; i++)
		base_anchor[i] = (float*)malloc(sizeof(float)* 4);

	generate_basic_anchors(anchor_widths, anchor_heights, anchor_num, basesize, base_anchor);

	//genrate all anchors
	for (int i = 0; i < f_h; i++)
	{
		for (int j = 0; j < f_w; j++)
		{
			float shift_x = j * feat_stride;
			float shift_y = i * feat_stride;

			for (int n = 0; n < anchor_num; n++)
			{
				anchors[(i*f_w + j)*anchor_num + n][0] = base_anchor[n][0] + shift_x;
				anchors[(i*f_w + j)*anchor_num + n][1] = base_anchor[n][1] + shift_y;
				anchors[(i*f_w + j)*anchor_num + n][2] = base_anchor[n][2] + shift_x;
				anchors[(i*f_w + j)*anchor_num + n][3] = base_anchor[n][3] + shift_y;
			}
		}
	}

	//free base_anchor
	for (int i = 0; i < anchor_num; i++) free(base_anchor[i]);
	free(base_anchor);
}
/*
    anchors;[][4]
*/
void generate_all_anchors(float **anchors,int batchsize, int f_w, int f_h, int feat_stride)
{

	int index = 0;
	int len = 10 * f_w*f_h;
	float **single_anchors = (float **)malloc(sizeof(float*)*len);
	for (int i = 0; i < len; i++)
		single_anchors[i] = (float*)malloc(sizeof(float)* 4);
	for (int i = 0; i < batchsize; i++)
	{
		generate_single_anchors(f_w, f_h, feat_stride, single_anchors);
		for (int j = 0; j < len; j++,index++)
		{
			anchors[index][0] = single_anchors[j][0];
			anchors[index][1] = single_anchors[j][1];
			anchors[index][2] = single_anchors[j][2];
			anchors[index][3] = single_anchors[j][3];
		}
	}
	//free single_anchors
	for (int i = 0; i < len; i++) free(single_anchors[i]);
	free(single_anchors);

}