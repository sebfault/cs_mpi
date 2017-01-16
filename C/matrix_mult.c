/*
Matrix Multiplication Parallel Computing
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define rows_a 2
#define rows_b 2
#define colms_a 2
#define colms_b 2

int main(int argc, char* argv[])
{

	int a[rows_a][colms_a];
	int b[colms_a][rows_b];
	int c[rows_a][colms_b];
	int rank, size, tag1, tag2, tag3,tag4;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	int i,j,k,offset,row;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;	

	tag1 = 1;
	tag2 = 2;
	tag3 = 3;
	tag4 = 4;

	printf("MPI task %d has started...\n", rank);
	row = rows_a / size;

	if(rank == MASTER){
		printf("Initializing arrays\n");
		for(i=0; i<rows_a; i++){
			for(j=0;j<colms_a; j++){
				a[i][j] = i+j;
				printf("a = %d",a[i][j]);
				}
			}
		for(i=0; i<colms_a; i++){
			for(j=0; j<colms_b; j++){
				b[i][j] = i+j;
				printf("b = %d",b[i][j]);
				}
			}
		for(i=0;i<rows_a;i++)
			for(j=0;j<colms_b;j++)
				c[i][j]=0;	
		offset = 0;		
		row = rows_a/size;
		int proc;
		//Sending data to the workers
		for(proc = 1; proc<size; proc++){
			printf("Sending %d rows to task %d\n",row,rank);
			MPI_Send(&b,colms_a*colms_b,MPI_INT,proc,tag1,MPI_COMM_WORLD);
			MPI_Send(&offset,1,MPI_INT,proc,tag2,MPI_COMM_WORLD);
			MPI_Send(&a[offset][0],row*colms_a,MPI_INT,proc,tag3,MPI_COMM_WORLD);
			MPI_Send(&row,1,MPI_INT,proc,tag4,MPI_COMM_WORLD);
			offset = offset + row;
		}
		//recieving data from the workers
		for(proc = 1; proc<size; proc++){
			MPI_Recv(&offset,1,MPI_INT,proc,tag1,MPI_COMM_WORLD,&status);
			MPI_Recv(&row,1,MPI_INT,proc,tag2,MPI_COMM_WORLD,&status);
			MPI_Recv(&c[offset][0],row*colms_a,MPI_INT,proc,tag3,MPI_COMM_WORLD,&status);
		}
		//calculation for master
		int tmp = offset + row;
		for(i=offset;i<tmp;i++){
			for(j=0;j<colms_b;j++){
				for(k=0;k<colms_a;k++){
					c[i][j] += a[i][k]*b[k][j];
					}
			}
		}
		//results
		int count = 0;
		for(i=0;i<rows_a;i++){
			for(j=0;j<colms_b;j++){
				printf("Count = %d .... %d\n",count,c[i][j]);
				count++;
				}
		}
	}

	if(rank>MASTER){
		MPI_Recv(&b,colms_a*colms_b,MPI_INT,MASTER,tag1,MPI_COMM_WORLD,&status);
		MPI_Recv(&offset,1,MPI_INT,MASTER,tag2,MPI_COMM_WORLD,&status);
		MPI_Recv(&a[offset][0],row*colms_a,MPI_INT,MASTER,tag3,MPI_COMM_WORLD,&status);
		MPI_Recv(&row,1,MPI_INT,MASTER,tag4,MPI_COMM_WORLD,&status);
		
		int tmp = offset + row;
		for(i=offset;i<tmp;i++){
			for(j=0;j<colms_b;j++){
				for(k=0;k<colms_a;k++){
				c[i][j] += a[i][k]*b[k][j];
				}
			}
		}
		MPI_Send(&offset,1,MPI_INT,MASTER,tag1,MPI_COMM_WORLD);
		MPI_Send(&row,1,MPI_INT,MASTER,tag2,MPI_COMM_WORLD);
		MPI_Send(&c[offset][0],row*colms_a,MPI_INT,MASTER,tag3,MPI_COMM_WORLD);
		
	}

	MPI_Finalize();	
}

