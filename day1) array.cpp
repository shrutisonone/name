//Find the duplicate in an array of N+1 integers. 
1) sort array and then check for i and i+1 element if same then that would be duplicate element
2) create frequency array and then check frequency
3) slow and fast pointer and then check cycle

int findDuplicate(vector<int>& nums) 
{
        int s=nums[0];
        int f=nums[0];
        do{
            s=nums[s];
            f=nums[nums[f]];
        }while(s!=f);
        
        f=nums[0];
        while(s!=f)
        {
            s=nums[s];
            f=nums[f];
        }
        
        return s;
}
//Sort an array of 0’s 1’s 2’s without using extra space or sorting algo 
1) sort(a, a+n);
2) count number of 0's , 1's , & 2's and then print 0's, 1's & 2's
3) Dutch National Flag using low , mid & high

void sortColors(vector<int>& nums) {
        int low=0;
        int high=nums.size()-1;
        int mid=0;
        
        while(mid<=high){
            switch(nums[mid]){
                    
                case 0:
                        swap(nums[low++], nums[mid++]);
                        break;
                case 1:
                        mid++;
                        break;
                case 2: 
                        swap(nums[mid], nums[high--]);
                        break;
            }
        }
        
    }
//Repeating and Missing Number
1) sort the array and then check difference between consecutive number if 1 then continue if 0 then repeating and then missing one calculate sum and then check
2) sum of n numbers n(n+1)/2 and sum of the squares n(n+1)(2n+1)/6
   and then X-Y=certain value (where certain value is difference between sum of given array and the real array that needs to be) 
   and then X-Y=certain value (---||---)(just change it two squares)
   and then solve these above two Equations
3) Using XOR
   XOR the given array and (4, 3, 6, 2, 1, 1)
   4^3^6^2^1^1=3
   3^(1^2^3^4^5^6)=4
   X^Y=4
   1 0 0
   so either of the bits would be 1/0 or 1/0
   so second bit 
   then XOR with all required
   then create two buckets and then if bits are set keep it in one set and if the bits are not set put in another
   |    6  |       |     3  |
   |    5  |       |     2  |
   |    4  |       |     1  | 
   |    6  |       |     1  |
   |___ 4__|       |     1  |
                   |     2  |
        5          |____ 3__|
                          1
          
// C++ program to Find the repeating
// and missing elements

#include <bits/stdc++.h>
using namespace std;

/* The output of this function is stored at
*x and *y */
void getTwoElements(int arr[], int n,
					int* x, int* y)
{
	/* Will hold xor of all elements 
	and numbers from 1 to n */
	int xor1;

	/* Will have only single set bit of xor1 */
	int set_bit_no;

	int i;
	*x = 0;
	*y = 0;

	xor1 = arr[0];

	/* Get the xor of all array elements */
	for (i = 1; i < n; i++)
		xor1 = xor1 ^ arr[i];

	/* XOR the previous result with numbers 
	from 1 to n*/
	for (i = 1; i <= n; i++)
		xor1 = xor1 ^ i;

	/* Get the rightmost set bit in set_bit_no */
	set_bit_no = xor1 & ~(xor1 - 1);

	/* Now divide elements into two 
	sets by comparing a rightmost set
	bit of xor1 with the bit at the same 
	position in each element. Also, 
	get XORs of two sets. The two
	XORs are the output elements. 
	The following two for loops 
	serve the purpose */
	for (i = 0; i < n; i++) {
		if (arr[i] & set_bit_no)
			/* arr[i] belongs to first set */
			*x = *x ^ arr[i];

		else
			/* arr[i] belongs to second set*/
			*y = *y ^ arr[i];
	}
	for (i = 1; i <= n; i++) {
		if (i & set_bit_no)
			/* i belongs to first set */
			*x = *x ^ i;
                else
			/* i belongs to second set*/
			*y = *y ^ i;
	}
	/* *x and *y hold the desired
		output elements */
}
//Merge two sorted Arrays without extra space 
1) Take another array which is of the size of the sum of the other two arrays put all the elements in the other array and then sort that array
2) If we aren't using  any extra space then we can use the technique of insertion sort an then play with two array check if smaller then ok else swap
1 4 7 8 10
2 3 9

1 2 7 8 10  --->sort    1 2 7 8 10
4 3 9                   3 4 9

1 2 3 8 10  ---->sort   1 2 3 8 10
7 4 9                   4 7 9

	// Function to find next gap.
int nextGap(int gap)
{
    if (gap <= 1)
        return 0;
    return (gap / 2) + (gap % 2);
}
 
void merge(int* arr1, int* arr2, int n, int m)
{
    int i, j, gap = n + m;
    for (gap = nextGap(gap); 
         gap > 0; gap = nextGap(gap)) 
    {
        // comparing elements in the first array.
        for (i = 0; i + gap < n; i++)
            if (arr1[i] > arr1[i + gap])
                swap(arr1[i], arr1[i + gap]);
 
        // comparing elements in both arrays.
        for (j = gap > n ? gap - n : 0; 
             i < n && j < m;
             i++, j++)
            if (arr1[i] > arr2[j])
                swap(arr1[i], arr2[j]);
 
        if (j < m) {
            // comparing elements in the second array.
            for (j = 0; j + gap < m; j++)
                if (arr2[j] > arr2[j + gap])
                    swap(arr2[j], arr2[j + gap]);
        }
    }
}
//Kadane’s Algorithm 
1) 3 for loops two for iterating and one for keeping max can be reduced to 2 for loops by keeping max in the 2nd for loop O(n^2)
2) local max and global max O(n)

int maxSubArray(vector<int>& nums) {
        int ls=nums[0];
        int gs=nums[0];
        
        for(int i=1;i<nums.size();i++)
        {
            ls=max(nums[i], ls+nums[i]);
            gs=max(ls, gs);
        }
        return gs;
        
    }
int maxSubArray(vector<int>& nums) {
        int sum=0;
        int maxi=INT_MIN;
        for(auto it:nums){
            sum+=it;
            maxi=max(sum, maxi);
            if(sum<0)
            {
                sum=0;
            }
        }
        return maxi;
    }
//Merge Overlapping Subintervals
1) Apply Brute Force such that compare each interval with everyone and then merge accordingly
2) sort out the interval and then compare each with the next one and then merge it
vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>>mergedIntervals;
        if(intervals.size()==0){
            return mergedIntervals;
        }

        sort(intervals.begin(), intervals.end());
        vector<int>tempInterval=intervals[0];

        for(auto it:intervals){
            if(it[0]<=tempInterval[1]){
                tempInterval[1]=max(it[1], tempInterval[1]);
            }
            else{
                mergedIntervals.push_back(tempInterval);
                tempInterval=it;
            }
        }
        mergedIntervals.push_back(tempInterval);
        return mergedIntervals;
    }

//Set Matrix Zeros 
1)take one more 2d matrix and then make the respective positions as zero as per zeros in the given matrix mark the zeros in the resultant matrix
2)take a 1d array to keep track of the zeros in the particular index and mark that zero in the respective extra row and column matrix and then after that reflect it
 in the 2d matrix
3) don't take the extra arrays for row and column but instead keep the first row and column to keep track and to avoid mistake use 
   col =false variable to keep track that it doesn't mark the extra zeros and remember to traverse from the last for  the marking
	
void setZeroes(vector<vector<int>>& matrix) {
int col0=1;
int rows=matrix.size();
int cols=matrix[0].size();

for(int i=0;i<rows;i++){
    if(matrix[i][0]==0)col0=0;
    for(int j=1;j<cols;j++)
        if(matrix[i][j]==0)
            matrix[i][0]=matrix[0][j]=0;
}

for(int i=rows-1;i>=0;i--){
    for(int j=cols-1;j>=1;j--)
        if(matrix[i][0]==0 || matrix[0][j]==0)
            matrix[i][j]=0;
        if(col0==0)matrix[i][0]=0;
}
}
	
//Pascal Triangle
1) calculate r-1
		C
		 c-1 to get the value of the particular index
2) If need to calculate a sequence of line then e.g for the 4th row 4   4   4      ........
								     C   C   C
								      0   1    2

vector<vector<int>> generate(int numRows) 
{
        vector<vector<int>>r(numRows);
        for(int i=0;i<numRows;i++)
        {
            r[i].resize(i+1);//Number of Rows =Number of Columns
            r[i][0]=r[i][i]=1;
            
            for(int j=1;j<i;j++)
                r[i][j]=r[i-1][j-1]+r[i-1][j];
        }
        return r;
}
	
//Inversion of an Array
1) brute force approach first compare the elements that is arr[i]>arr[i+1] and then check for the indices i<i+1 it would be
2) We can perform the Merge Sort which would give better time complexity in O(nlogn)
   and while performing merge sort check for the partition that after sorting the divided part then all the elemnets would be just greater after that
   // C++ program to Count 
// Inversions in an array 
// using Merge Sort 
#include <bits/stdc++.h> 
using namespace std; 

int _mergeSort(int arr[], int temp[], 
				int left, int right); 
int merge(int arr[], int temp[], int left, 
				int mid, int right); 

/* This function sorts the 
input array and returns the 
number of inversions in the array */
int mergeSort(int arr[], int array_size) 
{ 
	int temp[array_size]; 
	return _mergeSort(arr, temp, 0, array_size - 1); 
} 

/* An auxiliary recursive function 
that sorts the input array and 
returns the number of inversions in the array. */
int _mergeSort(int arr[], int temp[], 
				int left, int right) 
{ 
	int mid, inv_count = 0; 
	if (right > left) { 
		/* Divide the array into two parts and 
		call _mergeSortAndCountInv() 
		for each of the parts */
		mid = (right + left) / 2; 

		/* Inversion count will be sum of 
		inversions in left-part, right-part 
		and number of inversions in merging */
		inv_count += _mergeSort(arr, temp, 
								left, mid); 
		inv_count += _mergeSort(arr, temp, 
							mid + 1, right); 

		/*Merge the two parts*/
		inv_count += merge(arr, temp, left, 
						mid + 1, right); 
	} 
	return inv_count; 
} 

/* This funt merges two sorted arrays 
and returns inversion count in the arrays.*/
int merge(int arr[], int temp[], int left, 
		int mid, int right) 
{ 
	int i, j, k; 
	int inv_count = 0; 

	i = left; /* i is index for left subarray*/
	j = mid; /* j is index for right subarray*/
	k = left; /* k is index for resultant merged subarray*/
	while ((i <= mid - 1) && (j <= right)) { 
		if (arr[i] <= arr[j]) { 
			temp[k++] = arr[i++]; 
		} 
		else { 
			temp[k++] = arr[j++]; 

			/* this is tricky -- see above 
			explanation/diagram for merge()*/
			inv_count = inv_count + (mid - i); 
		} 
	} 

	/* Copy the remaining elements of left subarray 
(if there are any) to temp*/
	while (i <= mid - 1) 
		temp[k++] = arr[i++]; 

	/* Copy the remaining elements of right subarray 
(if there are any) to temp*/
	while (j <= right) 
		temp[k++] = arr[j++]; 

	/*Copy back the merged elements to original array*/
	for (i = left; i <= right; i++) 
		arr[i] = temp[i]; 

	return inv_count; 
} 

//Stock Buy And Sell
1) Brute Force
2)
 int maxProfit(vector<int>& prices) {
        int maxPro=0;
        int minPrice=INT_MAX;
        for(int i=0;i<prices.size();i++)
        {
            minPrice=min(minPrice, prices[i]);
            maxPro=max(maxPro, prices[i]-minPrice);
        }
        return maxPro;
    }

//Rotate Matrix
1) Take Another Matrix and then put into it as if the output but it would need extra space 
2) Transpose and then reverse column
void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();

        for(int i=0;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        for(int i=0;i<n;i++)
        {
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
//Search in 2D Matrix
1) Simple Brute Force approach just compare all the elements in matrix by iterating
2) Binary Search start from end of the first Row i.e. the last element of the first row
 bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int n=matrix.size();
        int m=matrix[0].size();
        int i=0;
        int j=m-1;

        while(i<n && j>=0)
        {
            if(matrix[i][j]==target){
               // cout<<"Found at"<<i<<", "<<j;
                return 1;
            }
            if(matrix[i][j]>target)
            j--;
            else
            i++;
        }
    }
    
3) Use the technique to find the index of the matrix and use smart binary search
bool searchMatrix(vector<vector<int>>& matrix, int target) {
        
        if(!matrix.size())return false;
        int n=matrix.size();
        int m=matrix[0].size();

        int lo=0;
        int hi=(n*m)-1;

        while(lo<=hi)
        {
            int mid=(lo+(hi-lo)/2);
            if(matrix[mid/m][mid%m]==target){
            return true;
            }
            
            if(matrix[mid/m][mid%m]<target){
            lo=mid+1;
            }
            else{
            hi=mid-1;
            }
        }
        return false;
            }
//pow(x, n)
1) just multiply the number by number of times the power is till it becomes zero so the complexity would become O(n)
2) If n is even then x=x*x and then n=n/2
   If n is odd then ans=ans*x and then n=n-1
double myPow(double x, int n) {
        double ans=1.0;
        long long nn=n;
        if(nn<0) nn=-1*nn;

        while(nn)
        {
            if(nn%2){
                ans=ans*x;
                nn=nn-1;
            }
            else
            {
                x=x*x;
                nn=nn/2;
            }
        }
        if(n<0)
        ans=(double)(1.0) / (double)(ans);
        return ans;
    }
//Majority Element >n/2 times
1) Brute Force O(n^2)
2) Hashing or Frequency Array then in O(n) but extra space so more optimization needed
3) Moore Voting Algorithm intution is frequency of a majority element is equal to the minority element
 int majorityElement(vector<int>& nums) {
        int count=0;
        int candidate =0;
        for(int num: nums){
            if(count==0){
                candidate=num;
            }
            if(num==candidate) count+=1;
            else count-=1;
        }
        return candidate;
    }
 // Majority Element >n/3 times
 optimized one using Boyer Moore 
 vector<int> majorityElement(vector<int>& nums) {
        int sz=nums.size();
        int num1=-1, num2=-1,count1=0, count2=0,i;
        for(i=0;i<sz;i++)
        {
            if(nums[i]==num1)
            count1++;
            else if (nums[i]==num2)
            count2++;
            else if(count1==0)
            {
                num1=nums[i];
                count1=1;
            }
            else if(count2==0)
            {
                num2=nums[i];
                count2=1;
            }
            else
            {
                count1--;
                count2--;
            }
        }
        vector<int>ans;
        count1=count2=0;
        for(i=0;i<sz;i++)
        {
            if(nums[i]==num1)
            count1++;
            else if(nums[i]==num2)
            count2++;
        }
        if(count1>sz/3)
        ans.push_back(num1);
        if(count2>sz/3)
        ans.push_back(num2);
        return ans;
    }
