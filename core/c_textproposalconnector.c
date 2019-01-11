/*
    date:2019 01 04
    author:hengk
    email:hengk@foxmail.com
    moudle: generarte text proposals
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct node{
	int index;
	struct node *next;
	struct node *right;

}node;

typedef struct box{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	struct box *next;
}box;

float min1(float a, float b)
{
	return a > b ? b : a;
}
float max1(float a, float b)
{
	return a > b ? a : b;
}
int meet_v_iou(int index1, int index2, float text_proposals[][4])
{
	float h = text_proposals[index1][3] - text_proposals[index1][1]+1;
	float h1 = text_proposals[index2][3] - text_proposals[index2][1]+1;

	float y = max1(text_proposals[index1][1], text_proposals[index2][1]);
	float y1 = min1(text_proposals[index1][3], text_proposals[index2][3]);

	float overlap = max1(0, y1 - y + 1) / min1(h, h1);

	float similarity = min1(h, h1) / max1(h, h1);

	if (overlap >= 0.7 && similarity >= 0.7)
		return 1;
	return 0;
}

node *get_precursors(int index, node* boxes_table, float text_proposals[][4], float score[], int len, int img_w)
{
	float *box = text_proposals[index];
	node * ret = (node*)malloc(sizeof(node));
	ret->index = -1;
	ret->right = NULL;
	for (int start = box[0] - 1; start >= (int)max1(box[0] - 50, 0); start--)
	{
		node  head = boxes_table[start], *cur = &head;
		while (cur && cur->index != -1)
		{
			if (meet_v_iou(cur->index, index, text_proposals))
			{
				//add a new node to the list
				if (ret->index == -1)
				{
					ret->index = cur->index;
				}
				else
				{
					node *n = (node*)malloc(sizeof(node));
					n->index = cur->index;
					n->right = ret->right;
					ret->right = n;
				}
			}
			cur = cur->right;
		}
		if (ret->index != -1)
			return ret;
	}
	return ret;
}

//judge the succession index is the right one or not
int is_successions(int index, int succession_index, node *boxes_table, float text_proposals[][4], float score[], int len, int img_w)
{
	node *ret = get_precursors(succession_index, boxes_table, text_proposals, score, len, img_w);
	node *precursors = ret;
	//get the max index 
	float max = 0;
	int max_index = -1;
	while (precursors  && precursors->index != -1){
		if (score[precursors->index]>max)
		{
			max = score[precursors->index];
			max_index = precursors->index;
		}
		precursors = precursors->right;
	}

	//free ret's memory
	while (ret)
	{
		node *tmp = ret;
		ret = ret->right;
		free(tmp);
	}

	if (score[index] >= score[max_index])
		return 1;
	return 0;
}

node *get_successions(int index, node* boxes_table, float text_proposals[][4], float score[], int len, int img_w)
{
	float *box = text_proposals[index];
	node * ret = (node*)malloc(sizeof(node));
	ret->index = -1;
	ret->right = NULL;
	for (int start = box[0] + 1; start < (int)min1(box[0] + 50, img_w); start++)
	{
		node  head = boxes_table[start], *cur = &head;
		while (cur && cur->index != -1)
		{
			if (meet_v_iou(cur->index, index, text_proposals))
			{
				//add a new node to the list
				if (ret->index == -1)
				{
					ret->index = cur->index;
				}
				else
				{
					node *n = (node*)malloc(sizeof(node));
					n->index = cur->index;
					n->right = ret->right;
					ret->right = n;
				}
			}
			cur = cur->right;
		}
		if (ret->index != -1)
			return ret;
	}
	return ret;


}


//generate the directed graph
void gen_graph(float text_proposals[][4], float score[], int len,
	int img_h, int img_w, unsigned char **graph)
{
	//create boxes table and initialize it
	node * boxes_table = (node*)malloc(img_w*(sizeof(node)));
	for (int i = 0; i < img_w; i++)
	{
		boxes_table[i].index = -1;
		boxes_table[i].right = NULL;
	}

	//put text_proposals and score index into boxes_table
	for (int i = 0; i < len; i++)
	{
		int x = text_proposals[i][0];
		//first time to insert
		if (boxes_table[x].index == -1)
		{
			boxes_table[x].index = i;
		}
		else
		{
			//insert new node after head node 
			node *n = (node*)malloc(sizeof(node));
			n->index = i;
			n->right = boxes_table[x].right;
			boxes_table[x].right = n;
		}

	}

	for (int i = 0; i < len; i++)
	{
		node *ret = get_successions(i, boxes_table, text_proposals, score, len, img_w);
		node *successions = ret;
		//get the max index 
		float max = 0;
		int max_index = -1;
		while (successions  && successions->index != -1){
			if (score[successions->index]>max)
			{
				max = score[successions->index];
				max_index = successions->index;
			}
			successions = successions->right;
		}
		if (max_index == -1) continue;

		if (is_successions(i, max_index, boxes_table, text_proposals, score, len, img_w))
		{
			graph[i][max_index] = 1;
		}


		//free the ret's memory
		while (ret){
			node* tmp = ret;
			ret = ret->right;
			free(tmp);
		}

	}

	//free the boxes_table's memory
	for (int i = 0; i < img_w; i++)
	{
		node *n = boxes_table[i].right;
		while (n){
			node *tmp = n;
			n = n->right;
			free(tmp);
		}
	}
	free(boxes_table);

}
void sub_graphs_connected(unsigned char **graph, int len, node *sub_graphs)
{
	node *cur = sub_graphs;
	for (int i = 0; i < len; i++)
	{
		//find the node which has no pre node 
		int j = 0;
		while (j<len && graph[j][i] == 0) j++;
		if (j < len)
			continue;
		//judge the node whether has next node
		j = 0;
		while (j<len && graph[i][j] == 0) j++;
		if (j == len)
			continue;

		//add the first node to sub_graphs
		if (cur->index == -1){
			cur->index = i;
		}
		else
		{
			node *n = (node *)malloc(sizeof(node));
			n->index = i;
			n->next = NULL;
			n->right = NULL;
			cur->next = n;
			cur = n;
		}

		//judge the node whether it includes next node
		int k = i;
		node *rcur = cur;
		while (rcur->right) rcur = rcur->right;
		while (1)
		{
			j = 0;
			while (j<len && graph[k][j] == 0) j++;
			if (j == len)
				break;
			k = j;
			node *n = (node*)malloc(sizeof(node));
			n->index = j;
			n->right = NULL;
			rcur->right = n;
			rcur = n;
		}


	}
}

// use anchors proposals to generate the text proposals
void get_text_lines(float text_proposals[][4], float score[], int len, int img_h, int img_w, float rbox_sco[][9], int *rlen)
{
	//create a two-dimension array dynamically (len*len)
	unsigned char **graph = (unsigned char**)malloc(len*sizeof(unsigned char*));
	for (int i = 0; i < len; i++)
	{
		graph[i] = (unsigned char*)malloc(len*sizeof(unsigned char));
		memset(graph[i], 0, len);
	}
	gen_graph(text_proposals, score, len, img_h, img_w, graph);
	/*int h = 0,h1=0;
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			if (graph[i][j] == 1)
				h++;
			else if (graph[i][j] == 0)
				h1++;

		}
	}
	printf("%d %d\n",h,h1);
	return;*/
	//crate a cross list to store sub_graphs
	node *sub_graphs = (node*)malloc(sizeof(node));
	sub_graphs->index = -1;
	sub_graphs->right = NULL;
	sub_graphs->next = NULL;
	sub_graphs_connected(graph, len, sub_graphs);


	node *cur = sub_graphs;
	//int m = 0;
	//while (cur)
	//{
	//	cur = cur->next;
	//	m++;
	//}
	//printf("%d\n",m);
	//get the final proposal_boxes
	box *final_proposal_boxes = (box *)malloc(sizeof(box));
	final_proposal_boxes->x1 = -1;
	final_proposal_boxes->y1 = -1;
	final_proposal_boxes->x2 = -1;
	final_proposal_boxes->y2 = -1;
	final_proposal_boxes->next = NULL;

	box * b = final_proposal_boxes;

	while (cur && cur->index != -1)
	{
		node *lcur = cur;
		float y1 = text_proposals[lcur->index][1];
		float y2 = text_proposals[lcur->index][3];
		float x1 = text_proposals[lcur->index][0];
		float x2 = text_proposals[lcur->index][2];
		int num = 0;
		float sco = 0;
		while (lcur){
			sco += score[lcur->index];
			//y1 += text_proposals[lcur->index][1];
			//y2 += text_proposals[lcur->index][3];
			
			if (text_proposals[lcur->index][0] < x1)
				x1 = text_proposals[lcur->index][0];
			
			if (text_proposals[lcur->index][2]>x2)
				x2 = text_proposals[lcur->index][2];

			if (text_proposals[lcur->index][1] < y1)
				y1 = text_proposals[lcur->index][1];
			
			if (text_proposals[lcur->index][3]>y2)
				y2 = text_proposals[lcur->index][3];
			lcur = lcur->right;
			num++;
		}
		sco = sco / num;
		int m = (x2 - x1) / num * 0.5;
		x1 = x1 - m;
		x2 = x2 + m;

		if (b->x1 == -1 && b->x2 == -1 && b->y1 == -1 && b->y2 == -1)
		{
			b->x1 = x1;
			b->y1 = y1;
			b->x2 = x2;
			b->y2 = y2;
			b->score = sco;

		}
		else
		{
			box *n_b = (box*)malloc(sizeof(box));
			n_b->x1 = x1;
			n_b->y1 = y1;
			n_b->x2 = x2;
			n_b->y2 = y2;
			n_b->score = sco;
			n_b->next = NULL;
			b->next = n_b;
			b = n_b;

		}
		cur = cur->next;
	}

	//clip the box
	b = final_proposal_boxes;
	int index = 0;
	while (b)
	{
		b->x1 = min1(max1(0, b->x1), img_w - 1);
		b->y1 = min1(max1(0, b->y1), img_h - 1);
		b->x2 = min1(max1(0, b->x2), img_w - 1);
		b->y2 = min1(max1(0, b->y2), img_h - 1);
		if (b->score <= 0.7 && (b->x2-b->x1)/(b->y2-b->y1)<=0.5 && (b->x2-b->x1)<16)
		{
			b = b->next;
			continue;
		}
		rbox_sco[index][0] = b->x1;
		rbox_sco[index][1] = b->y1;

		rbox_sco[index][2] = b->x2;
		rbox_sco[index][3] = b->y1;

		rbox_sco[index][4] = b->x1;
		rbox_sco[index][5] = b->y2;

		rbox_sco[index][6] = b->x2;
		rbox_sco[index][7] = b->y2;

		rbox_sco[index][8] = b->score;
		//printf("%lf %lf %lf %lf %lf\n", b->x1, b->y1, b->x2, b->y2, b->score);
		index++;
		b = b->next;
	}
	*rlen = index;

	//free graph's memory
	for (int i = 0; i < len; i++) free(graph[i]);
	free(graph);
	//free sub_graphs' memory
	while (sub_graphs){
		node *cur = sub_graphs;
		sub_graphs = sub_graphs->next;
		while (cur) {
			node *tmp = cur;
			cur = cur->right;
			free(tmp);
		}
	}
	//free boxes
	while (final_proposal_boxes){
		box *n = final_proposal_boxes;
		final_proposal_boxes = final_proposal_boxes->next;
		free(n);
	}
}

//int main()
//{
//	float text_proposals[][4] = { 1, 1, 4, 4, 5, 1, 5, 4 ,50,50,55,55,60,50,65,55};
//	float a[][9] = { 1, 2, 3, 4, 5,6,7,8,9,
//		1,2,3,4,5,6, 7, 8, 9 };
//	float score[] = {0.9,0.8,0.7,0.9};
//	int len = 4;
//	int len1 = 3;
//	get_text_lines(text_proposals, score, len, 100, 200,a,&len1);
//
//	return 1;
//}