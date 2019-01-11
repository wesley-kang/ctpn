/*
    date:2019 01 08
    author:hengk
    email:hengk@foxmail.com
    moudle: handle network output
*/
#include <stdio.h>
#include <stdlib.h>
void generate_all_anchors(float **anchors, int batchsize, int f_w, int f_h, int feat_stride);
void bbox_transform_inv(float **boxes, float **deltas, float **proposals, int len);
void nms(float **dets, int len, float thresh, int inds[], int *inds_len);
float max1(float a, float b);
float min1(float a, float b);
/*
    proprosals: [][5]
*/
void bubble_sort(float **proprosals, int len)
{
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len-i-1; j++)
		{
			if (proprosals[j][4] < proprosals[j + 1][4])
			{

			        float *tmp = proprosals[j];
			        proprosals[j] = proprosals[j+1];
				    proprosals[j + 1] = tmp;

			}
		}
	}
}
/*
    cls : [][2]
    box_deltas:[][4]
*/
void generate_proposals(float **cls,float **box_deltas,int len,int batchsize,int fea_w,
								int fea_h,int width,int height,int feat_stride,float **ret,int *rrlen)
{
	int pre_nms_topn = min1(5000,fea_w*fea_h*10);
	int	post_nms_topn = 2000;
	float	nms_thresh = 0.3;
	int	min_size = 4;
	int	num_anchors = 10;

	float **proprosals = (float**)malloc(sizeof(float*)*len);
	float **anchors = (float**)malloc(sizeof(float*)*len);
	float **box_score = (float**)malloc(sizeof(float*)*len);
	for (int i = 0; i < len; i++)
	{
		box_score[i] = (float*)malloc(sizeof(float)* 5);
		proprosals[i] = (float*)malloc(sizeof(float)*4);
		anchors[i] = (float*)malloc(sizeof(float)* 4);
	}
	int *index = (int*)malloc(sizeof(int)*pre_nms_topn);
	int rlen = pre_nms_topn;

	generate_all_anchors(anchors, batchsize, fea_w, fea_h, feat_stride);
	bbox_transform_inv(anchors, box_deltas, proprosals, len);

	for (int i = 0; i < len; i++)
	{

		box_score[i][0] = proprosals[i][0];
		box_score[i][1] = proprosals[i][1];
		box_score[i][2] = proprosals[i][2];
		box_score[i][3] = proprosals[i][3];
		box_score[i][4] = cls[i][1];

	}

	int ids = 0;
	for (int i = 0; i < batchsize; i++)
	{

		float **tmp = (box_score + i*(fea_h*fea_w * 10));

		bubble_sort(tmp, fea_h*fea_w * 10);

		nms(tmp,pre_nms_topn, nms_thresh, index, &rlen);
		for (int j = 0; j < (int)min1(rlen,post_nms_topn); j++)
		{
			ret[ids][0] = i;
			ret[ids][1] = tmp[index[j]][4];
			ret[ids][2] = min1(max1(tmp[index[j]][0],0),width);
			ret[ids][3] = min1(max1(tmp[index[j]][1],0),height);
			ret[ids][4] = min1(max1(tmp[index[j]][2],0),width);
			ret[ids][5] = min1(max1(tmp[index[j]][3],0),height);
			int h = ret[ids][5] - ret[ids][3];
			int w = ret[ids][4] - ret[ids][2];
			if (h >= min_size && w >= min_size)
			{
				ids++;
			}

		}


	}
    *rrlen = ids;
	//free proprosals and anchors
	for (int i = 0; i < len; i++)
	{
		free(box_score[i]);
		free(proprosals[i]);
		free(anchors[i]);
	}
	free(index);
	free(box_score);
	free(proprosals);
	free(anchors);

}