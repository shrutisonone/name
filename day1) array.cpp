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
create the frequency array mark the element which are present then one of the element will have frequency 2 and another one woukd have 0 that would be repeating and missing number
1) (do not say this cause it would modify the array) sort the array and then check difference between consecutive number if 1 then continue if 0 then repeating and then missing one calculate sum and then check
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

//Next Permutation
1) next_permutation(begin(nums), end(nums));
2) 1 3 5 4 2 --> 1 4 2 3 5
	travserse from back and find A[i]<A[i+1]	{{{{{index1=1}}}}
	travserse from back and find value greater than index1 value A[index2]>A[index1]    {{{{{index2=3}}}}
	swap the value at index1 and index2 swap(A[index1], A[index2])
		so the array would look like	1 4 5 3 2
	reverse all the elements from index1 to the last element i.e. reverse(index1, nums.end())
#include<bits/stdc++.h>
using namespace std;

int main()
{
    int testcase;
    cin>>testcase;
    while(testcase--)
    {
        int size;
        cin>>size;
        vector<int>nums(size);
        for(int i=0;i<size;i++)
        {
            cin>>nums[i];
        }
        
         int n=nums.size(), k, l;
        for(k=n-2;k>=0;k--)
        {
            if(nums[k]<nums[k+1])
                break;
        }
        
        if(k<0)
        {
            reverse(nums.begin(), nums.end());
        }
        else
        {
            for(l=n-1;l>k;l--)
            {
                if(nums[l]>nums[k])
                    break;
            }
            swap(nums[k], nums[l]);
            reverse(nums.begin()+k+1, nums.end());
        }
        for(int i=0;i<n;i++)
        {
            cout<<nums[i]<<" ";
        }
    }
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
    
//Unique Path
1) recursive call for bottom and Right Index and for optimal use memoziation problem
int countpaths(int i, int j, int n, int m)
{
    if(i==(n-1) && j==(m-1))
    return 1;
    if(i>=n || j>=m)
    return 0;
    else
    return countpaths(i+1, j)+countpaths(i, j+1);
}
// using dp

int countpaths(int i, int j, int n, int m)
{
    if(i==(n-1) && j==(m-1))
    return 1;
    if(i>=n || j>=m)
    return 0;
    if(dp[i][j]!=-1) 
	    return dp[i][j];
    else 
	    return dp[i][j]=countpaths(i+1, j, dp)+countpaths(i, j+1, dp);
}
//Efficent using ncr 
//Total blocks to be traversed is (n+m-2) and then 
(n+m-2)         (n+m-2)
	C              C
	 n-1            m-1
		
 int uniquePaths(int m, int n) {
        int N=n+m-2;
        int r=m-1;

        double res=1;

        for(int i=1;i<=r;i++)
            res=res* (N-r+i)/i;

        return (int)res;
    }


//2 Sum
//1) Sort the array and then have two for loops and then check if the sum of the two number is equal to target sum if there then print the i and j value and break O(n^2)
//2) Sort Take one pointer to left and one to right and then check if the sum is equal to target sum or not if sum is less than target sum
// then move the left pointer forward and if the sum is greater than target sum move right pointer backward6
//3) using Hashing in O(n) take thefirst number if the (target sum-nums[i]) is there in the hashmap then return both index else traverse till the end of array
vector<int> twoSum(vector<int>& nums, int target) {
        vector<int>ans;
        unordered_map<int, int>mp;
        
        for(int i=0;i<nums.size();i++){
            if(mp.find(target-nums[i])!=mp.end()){
                ans.push_back(mp[target-nums[i]]);
                ans.push_back(i);
                return ans;
            }
            mp[nums[i]]=i;
        }
        return ans;
    }

//4 sum problem
//1) brute force O(n^4)
//2) sort and then use hashmap and then check like 2 sum but O(n^3)+O(nlogn) and space O(n)
//3) sort take i and j put j=i+1 and then use left and right pointer for getting the value of remaining sum O(n^3) and space O(1)
 vector<vector<int>> fourSum(vector<int>& num, int target) {
        vector<vector<int>>res;
        if(num.empty()) return res;


        int n=num.size();
        sort(num.begin(), num.end());

        for(int i=0;i<n;i++)
        {
            for(int j=i+1;j<n;j++)
            {
                int target_2=target-num[j]-num[i];
                int front=j+1;
                int back=n-1;

                while(front < back)
                {
                    int two_sum=num[front]+num[back];
                    if(two_sum<target_2) front++;
                    else if(two_sum>target_2) back--;
                    else
                    {
                        vector<int>quadruplet(4, 0);
                        quadruplet[0]=num[i];
                        quadruplet[1]=num[j];
                        quadruplet[2]=num[front];
                        quadruplet[3]=num[back];
                        res.push_back(quadruplet);

                        while(front<back && num[front]==quadruplet[2]) ++front;
                        while(front<back && num[back]==quadruplet[3]) --back;
                    }
                }
                while(j+1<n && num[j+1]==num[j]) ++j;
            }
            while(i+1<n && num[i+1]==num[i]) ++i;
        }
        return res;
    }

//Longest Consecutive Sequence 
//Sort the array and then just check the difference if 1 is the difference then increment the count and iterate further and keep track of count
//Efficient is to put that all in the hashset and then iterate over the array and check ofr the one lesser than it 
e.g 102 1 100 101 2 3 4
	//then if 102 then if 101 exist then do not do any thing if not exist consider it as root and then and then increment by one and then keep track of count
	 int longestConsecutive(vector<int>& nums) {
        
        set<int>hashset;
        for(int num:nums)
        {
            hashset.insert(num);  //insert every element into the hashset
        }
        
        int cnt=0;
        for(int num : nums){
            if(!hashset.count(num-1))  //if less than that number does not exist then we got the root
            {
                int currnum=num;
                int currcnt=1;
                
                while(hashset.count(currnum+1))  //check for the further series ans then increment value of currnum and currcnt
                {
                    currnum++;
                    currcnt++;
                }
                cnt=max(cnt, currcnt);
            }
        }
        return cnt;
    }

//Largest subarray with 0 sum 
//just run the two for loop and then check if the sum is zero  but n^2 so use map to reduce to O(n)
int maxLen(int arr[], int n) 
{ 
    // Initialize result 
    int max_len = 0;  
  
    // Pick a starting point 
    for (int i = 0; i < n; i++) { 
  
        // Initialize currr_sum for 
        // every starting point 
        int curr_sum = 0; 
  
        // try all subarrays starting with 'i' 
        for (int j = i; j < n; j++) { 
            curr_sum += arr[j]; 
  
            // If curr_sum becomes 0,  
            // then update max_len 
            // if required 
            if (curr_sum == 0) 
                max_len = max(max_len, j - i + 1); 
        } 
    } 
    return max_len; 
} 
int maxLen(int a[], int n)
{
    // Your code here
    map<int, int>mp;
    int maxi=0;
    int sum=0;
    
    for(int i=0;i<n;i++)
    {
        sum+=a[i];
        if(sum==0)
        {
            maxi=i+1;
        }
        else
        {
            if(mp.find(sum)!=mp.end())
            {
                maxi=max(maxi, i-mp[sum]);
            }
            else
            {
                mp[sum]=i;
            }
        }
    }
    return maxi;
}
//Count the number of subarrays having a given XOR
//Navie Approach is to take two fro loops and then compare the xor is equal to given and then increment count in O(n^2)
//Using Hashmap store that value i.e. similar to prefix XOR and then check for count of Y if Y ^ K=XR(XOR for prefix array) 

[-------XR------------]	
----------------------
[----Y---][----K-----]

//so count number of time Y that would be count of the subarray

int Subarray_XOR(vector<int>&arr, int x)
{
	map<int , int>mp;
	int cnt=0;
	int xor=0;
	for(auto it:arr)
	{
		xor=xor^it;
		
		if(xor==x)
		{
			cnt++;
		}
		if(mp.find(xor^x)!=mp.end())
		{
			cnt+=mp[xor^x];
		}
		mp[xor]+=1;
	}
	return cnt;
}

//Longest substring without repeat 
//Navie approach is using two for loops and then check if any repeating charcter it would take O(n^3)  ----further optimized-->o(n^2)
//O(2n) uisng set take two pointers left and right and then check if the current char exist in set or not if it does not exist put it in set and increment right pointer 
//and if the current elemnet exist then increment the left pointer one by one upto the till that repeating character is not eleminated so to more optimized this solution use 
//map that would store the element with index to keep track of the index so instead incrementing one by one the left pointer directly jump to the direct non repeating character
 int lengthOfLongestSubstring(string s) {
        
        vector<int>mp(256, -1);
        
        int left=0;
        int right=0;
        int n=s.size();
        int cnt=0;
        
        while(right<n){
            if(mp[s[right]]!=-1)
            {
                left=max(left, mp[s[right]]+1);
            }
            mp[s[right]]=right;
            cnt=max(cnt, right-left+1);
            right++;
        }
        return cnt;
        
    }

//Reverse a Linked List

Node *reverse(Node *head)
{
	Node *a=head;
	Node *b=head;
	Node c=NULL;
	while(a!=NULL)
	{
		b=a->next;
		a->next=c;
		c=a;
		a=b;
	}
	return c;
}

*Recurrsively 

Node *reverse(Node *head)                                 1	1     
{							  |	|____
	Node *curr=head;				  2 	2____|
	if(curr==NULL || curr->next==NULL)		  |	|____
	return curr;					  3	3____|
							  |	|____
	Node *rest=reverse(curr->next);			  4	4____|
	curr->next->next=curr;				  |	|    |   //encircle one are reverse pointers i.e. curr->next->next=curr;
	curr->next=NULL;				  5	5____|	 //curr->next=NULL breaks the old links of the linked list
							  |	|____|
	return rest;					 NULL	NULL
}

//Find the middle of linked list
Navie solution to count the length and then divide by 2 and then handle the odd and even length accordingly i.e. according to question if length is even then which to retur first middle or second middle
//optimised is with fast and slow pointers move them and if fast reaches NULL or fasts next is null the break;

Node *findmiddle(Node *head)
{
	Node *f=head;
	Node *s=head;
	while(f!=NULL && f->next!=NULL)
	{
		f=f->next->next;
		s=s->next;
	}
	return s;
}
//Merge Two sorted linked list
Navie Solution is to take any data structure merge two into one and perform sort and then make a linked list of that sorted numbers
//Optimised is to take two pointer l1 and l2 make sure that at start they are pointing to head and then and also the l1 will always be pointing to the smaller one
//iterate till both l1 and l2 not equals to null and then inside take one more loop if l->val is greater in case then make tmp whihc is behind l1 make it to pint l2 and then swap l1 and l2

Node *mergetwosortedlist(Node *l1, Node *l2)
{
	if(l1==NULL)
	return l2;
	
	if(l2==NULL)
	return l1;
	
	if(l1->data > l2->data)
	swap(l1, l2);
	
	Node *res=l1;
	Node *tmp;
	
	while(l1!=NULL && l2!=NULL)
	{
		while(l1!=NULL && l1->data <= l2->data)
		{
			tmp=l1;
			l1=l1->next;
		}
		tmp->next=l2;
		swap(l1, l2);
	}
	return res;
}

//Remove nth node from end of linked list
Navie Approach is to traverse a linked list count the nodes find length and then subtract it that node from end will be the length-kth from start and then traverse till the erailer node and then change the link
//Optimised approach is to take two pointer fast and slow and then make fast pointer traverse first till n-1 and then travserse both fast and slow till fast next!=NULL and then change pointers

Node *deletenthnodefromend(Node *head, int n)
{
	Node *f=head;
	Node *s=head;
	
	for(int i=0;i<n;i++)
	{
		f=f->next;
		if(f==NULL)
		return head->next;
	}
	
	while(f->next!=NULL)
	{
		f=f->next;
		s=s->next;
	}
	s->next=s->next->next;
	return head;
}
//Add Two Numbers Respresented by linked list start from the last and then add %10 to the new node and if there exist a carry propogate it

Node *addtwolist(Node *l1, Node *l2)
{
	Node *res= new Node();
	Node *tmp=res;
	
	int carry=0;
	int sum=0;
	
	while(l1!=NULL || l2!=NULL || carry)
	{
		if(l1!=NULL){
			sum+=l1->data;
			l1=l1->next;
		}
		
		if(l2!=NULL){
			sum+=l2->dtaa;
			l2=l2->next;
		}
		sum+=carry;
		carry=sum/10;
		Node *a=new Node(sum%10);
		tmp->next=a;
		tmp=tmp->next;
	}
	return res->next;
}

/////////////Intersection of Y Shaped Linked List
One is two travserse through one and then check the other so n^2
Optimised is to calculate the difference and then allow the longer one to traverse that nodes eariler and then start travserse both
More Optimised Start both at one time and if any of them is NULL assign it to the start of anaother one
int intersectionYShappedLinkedLists(Node *head1, Node *head2)
{
	Node *p=head1;
	Node *q=head2;
	
	while(p!=q){
		
	p=p->next;
	q=q->next;
	
	if(p==NULL)
	p=head2;
	
	if(q==NULL)
	q=head1;
	}
	return q->data;
}

//Detect Loop in a linked list

bool DetectLoop(Node *head)
{
	Node *f=head;
	Node *s=head;
	
	while(f!=NULL && f->next!=NULL)
	{
		f=f->next->next;
		s=s->next;
		
		if(f==s)
		return true;
	}
	return false;
	
}
//detect and remove
 ListNode *detectCycle(ListNode *head) {
        ListNode *f=head;
        ListNode *s=head;
        
        while(f!=NULL && f->next!=NULL)
        {
            f=f->next->next;
            s=s->next;
            if(f==s)
                break;
        }
        
        if(f==NULL || f->next==NULL) return NULL;
        
            s=head;
            while(f!=s)
            {
                f=f->next;
                s=s->next;
            }
      //  cout<<f->val<<endl; start point
            return s;
        
       
    }

//Reverse a linked list in group of size k

Node *reverselinkedlistsizek(Node *head, int k)
{
	Node *a=head;
	Node *b=head;
	Node *c=NULL;
	int cnt=k;
	while(a!=NULL && cnt--)
	{
		b=a->next;
		a->next=c;
		c=a;
		a=b;
	}
	if(head!=NULL)
	head->next=reverselinkedlistsizek(b, k);
	
	return c;
	
//Check if palindrome
	but extra space O(n) stack
bool isPalindrome(Node *head)
{
    stack<int>s;
    Node *h1=head;
    while(h1!=NULL)
    {
        s.push(h1->data);
        h1=h1->next;
    }
    while(head!=NULL)
    {
        if(s.top()!=head->data)
        {
            return 0;
        }
        s.pop();
        head=head->next;
    }
    return 1;
}
//without using extra space

Node *reverse(Node *head)
{
    Node *a=head;
    Node *b=head;
    Node *c=NULL;
    while(a!=NULL)
    {
        b=a->next;
        a->next=c;
        c=a;
        a=b;
    }
    return c;
}
	
bool isPalindrome(Node *head)
{
    //Your code here
    Node *f=head;
    Node *s=head;
    while(f!=NULL && f->next!=NULL)
    {
        f=f->next->next;
        s=s->next;
    }
    f=head;
    s=reverse(s);
    while(s!=NULL)
    {
        if(f->data!=s->data)
        {
            return false;
        }
        f=f->next;
        s=s->next;
    }
    return true;
}


//Roatate a linked list by k
	Node* rotate(Node* head, int k)
    {
        // Your code here
       Node *a=head;
       
       while(a->next!=NULL)
       {
           a=a->next;
       }
       a->next=head;
       
       Node *b;
       while(k--)
       {
           b=head;
           head=head->next;
       }
       
       b->next=NULL;
       return head;
    }


//Flattening of a linked list
Node *mergeTwolist(Node *a, Node *b)
{
	Node *tmp=new Node(0);
	Node *res=tmp;
	
	while(a!=NULL && b!=NULL)
	{
		if(a->data<b->data)
		{
			tmp->bottom=a;
			tmp=tmp->bottom;
			a=a->bottom;
		}
		else
		{
			tmp->bottom=b;
			tmp=tmp->bottom;
			b=b->bottom;
		}
		
		if(a) tmp->bottom=a;
		else tmp->bottom =b;
		
		return res->bottom;
	}
}
Node *flatten(Node *root)
{
	if(root==NULL || root->next==NULL)
	return root;
	
	root->next=flatten(root->next);
	root->next=mergetwolist(root, root->next);
	
	return root;
}


//Clone a linked list
	
Node *copylist(Node *head)
{
	Node *itr=head;
	Node *front=head;
	
	while(itr!=NULL)
	{
		front=itr->next;
		Node *copy=new Node(itr->val);
		itr->next=copy;
		copy->next=front;
		itr=front;
	}
	
	itr=head;
	while(itr!=NULL)
	{
		if(itr->random!=NULL)
		{
			itr->next->random=itr->random->next;
		}
		itr=itr->next->next;
	}
	
	itr=head;
	Node *res=new Node(0);
	Node *copy=res;
	
	while(itr!=NULL)
	{
		front=it->next->next;
		copy->next=itr->next;
		copy=copy->next;
		itr=front;
	}
	return res->next;
}


//3 Sum 
//Navie Approach is to have three for loop and then add condition and then add the satisfying triplets in output vector O(n^3);
//Optimised is to use two loops and map to keep track then if there is remaining sum in the map then it adds upto the result
//More Optimised is to traverse the array and use two pointer approach so that complexity is maintained at O(n)
vector<vector<int>> threeSum(vector<int>& nums) {
        
        sort(nums.begin(), nums.end());
        vector<vector<int>>res;
        
        for(int i=0;i<(int)nums.size()-2;i++)
        {
            if(i==0 || (i>0 && nums[i]!=nums[i-1]))
            {
                int lo=i+1;
                int hi=nums.size()-1;
                int sum=0-nums[i];
                
                while(lo<hi){
                    if(nums[lo]+nums[hi]==sum){
                        vector<int>tmp;
                        tmp.push_back(nums[i]);
                        tmp.push_back(nums[lo]);
                        tmp.push_back(nums[hi]);
                        
                        res.push_back(tmp);
                        
                        while(lo<hi && nums[lo]==nums[lo+1]) lo++;
                        while(lo<hi && nums[hi]==nums[hi-1]) hi--;
                        
                        lo++;
                        hi--;
                    }
                    else if(nums[lo]+nums[hi] < sum) lo++;
                    else hi--;
                }
            }
        }
        return res;
    }
	
//Trapping Rain Water Problem ------------max from right max from left min of those minus the current height
//Navie Solution premax array and suffixmax array and then and then just take the min of two max and minus the value
//stack based of same complexity 
//Most Optimised is to use 2 pointer right max and left max to calculate the values

int trappingWater(int arr[], int n){

    // Your code here
    int res=0;
    int lmax=0;
    int rmax=0;
    
    int l=0;
    int h=n-1;
    
    while(l<=h)
    {
        if(arr[l]<arr[h])
        {
            if(arr[l]>lmax)
            lmax=arr[l];
            else
            res+=lmax-arr[l];
            
            l++;
        }
        else
        {
            if(arr[h]>rmax)
            rmax=arr[h];
            else
            res+=rmax-arr[h];
            
            h--;
        }
    }
    return res;
    
}
	
// Remove Duplicates from Sorted Array
	
use two pointer approach if only the value is different then swap else no swap and increment j and keep i as track for the last unique
 int removeDuplicates(vector<int>& nums) {
        if(nums.size()==0)
            return 0;
        int i=0,j=0;
        for( j=1;j<nums.size();j++)
        {
            if(nums[i]!=nums[j])
            {
                i++;
                nums[i]=nums[j];
            }
        }
        return i+1;
    }
	

	
//Max consecutive ones
if 0 cnt=0 else inceremnt count
 int findMaxConsecutiveOnes(vector<int>& nums) {
        int cnt=0;
        int maxi=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]==0)
            {
                cnt=0;
            }
            else
            {
                cnt++;
            }
            maxi=max(maxi, cnt);
        }
        return maxi;
    }
//N Meetings in Room
	struct meeting
{
    int start;
    int end;
    int pos;
};

bool comparator(struct meeting m1, struct meeting m2)
{
    if (m1.end < m2.end) return true; 
    else if(m1.end > m2.end) return false; 
    else if(m1.pos < m2.pos) return true; 
    return false;
}
class Solution
{
    public:
    //Function to find the maximum number of meetings that can
    //be performed in a meeting room.
    int maxMeetings(int s[], int e[], int n)
    {
        // Your code here
        struct meeting meet[n]; 
        for(int i = 0;i<n;i++)
        {
            meet[i].start = s[i], meet[i].end = e[i], meet[i].pos = i+1; 
        }
        
        sort(meet, meet+n, comparator); 
        
        vector<int> answer;
        
        int limit = meet[0].end; 
        answer.push_back(meet[0].pos); 
        
        for(int i = 1;i<n;i++) 
        {
            if(meet[i].start > limit)
            {
                limit = meet[i].end; 
                answer.push_back(meet[i].pos); 
            }
        }
        // for(int i = 0;i<answer.size();i++) {
        //     cout << answer[i] << " "; 
        // }
        return answer.size();
    }
};
	
	
//minimum platform needed
//just sort the arrival and departure array and then compare there trimmings
#include<bits/stdc++.h>
using namespace std;

int main()
{
    int testcase;
    cin>>testcase;
    while(testcase--)
    {
        int size;
        cin>>size;
        vector<int>arr(size);
        vector<int>dep(size);
        for(int i=0;i<size;i++)
        {
            cin>>arr[i];
        }
        for(int i=0;i<size;i++)
        {
            cin>>dep[i];
        }
        
        sort(arr.begin(), arr.end());
        sort(dep.begin(), dep.end());
        
        int i=1;
        int j=0;
        int maxi=1;
        int paltform=1;
        
        while(i<size && j<size)
        {
            if(arr[i]<=dep[j])
            {
                paltform++;
                i++;
            }
            else if(arr[i]>dep[j])
            {
                paltform--;
                j++;
            }
            maxi=max(maxi, paltform);
            
        }
        cout<<maxi<<endl;
    }
    return 0;
}

//Job Sequencing Problem
	
bool comp( Job j1, Job j2)
{
    return (j1.profit>j2.profit);
}

class Solution 
{
    public:
    //Function to find the maximum profit and the number of jobs done.
    vector<int> JobScheduling(Job arr[], int n) 
    { 
        // your code here
        sort(arr, arr+n, comp);
        int maxi=arr[0].dead;
        for(int i=1;i<n;i++)
        {
            maxi=max(maxi, arr[i].dead);
        }
        
        int slot[maxi+1];
        memset(slot, -1, sizeof(slot));
        
        int countjob=0;
        int jobprofit=0;
        
        
        for(int i=0;i<n;i++)
        {
            for(int j=arr[i].dead;j>0;j--)
            {
                if(slot[j]==-1)
                {
                    slot[j]=i;
                    countjob++;
                    jobprofit+=arr[i].profit;
                    break;
                }
            }
        }
        return {countjob, jobprofit};
    } 
};
	
//Fractional Knapsack value by weight ration P/W profit by weight greedy
	
bool comp(Item it1, Item it2)
{
    return (((double)it1.value/(double)it1.weight )> ((double)it2.value/(double)it2.weight));
}
class Solution
{
    public:
    //Function to get the maximum total value in the knapsack.
    double fractionalKnapsack(int W, Item arr[], int n)
    {
        // Your code here
        sort(arr, arr+n, comp);
        int currweight=0;
        double finalvalue=0.0;
        
        for(int i=0;i<n;i++)
        {
            if(currweight+arr[i].weight<=W)
            {
                currweight+=arr[i].weight;
                finalvalue+=arr[i].value;
            }
            else
            {
                int remain=W-currweight;
                finalvalue+=(arr[i].value/(double)arr[i].weight) *    ( (double)remain);
                break;
                
            }
        }
        return finalvalue;
    }
        
};

	
//Min Number of Coins Greedy
	
int findmincoins(int V)
{
    int deno={1, 2, 5, 10, 20, 50, 100, 500, 1000};
    int n=9;
    
    vector<int>ans;
    
    for(int i=n-1;i>=0;i--)
    {
        while(V>=deno[i]){
            V-=deno[i];
            ans.push_back(deno[i]);
        }
    }
    
    for(int i=0;i<ans.size();i++)
    {
        cout<<ans[i]<<" ";
    }
    cout<<"Min Number of Coins"<<ans.size()<<endl;
}
	
//Subset Sums
// 	Input:
// N = 2
// Arr = [2, 3]
// Output:
// 0 2 3 5
// Explanation:
// When no elements is taken then Sum = 0.
// When only 2 is taken then Sum = 2.
// When only 3 is taken then Sum = 3.
// When element 2 and 3 are taken then 
// Sum = 2+3 = 5.
	
	void func(int ind, int sum, vector<int>&arr,int N, vector<int>&SubsetSum)
    {
        if(ind==N)
        {
            SubsetSum.push_back(sum);
            return ;
        }
        func(ind+1, sum+arr[ind], arr, N,  SubsetSum);
        func(ind+1, sum, arr,N ,  SubsetSum);
    }
    vector<int> subsetSums(vector<int> arr, int N)
    {
        // Write Your Code here
        vector<int>SubsetSum;
        func(0, 0, arr, N, SubsetSum);
        sort(SubsetSum.begin(), SubsetSum.end());
        return SubsetSum;
    }
	
//subset sum 2 where find power set without duplicates
	
class Solution {
public:
    void func(int ind, vector<int>&nums, vector<int>&ds, vector<vector<int>>&ans)
    {
        ans.push_back(ds);
        for(int i=ind;i<nums.size();i++)
        {
            if(i!=ind && nums[i]==nums[i-1])    continue;
            ds.push_back(nums[i]);
            func(i+1, nums, ds, ans);
            ds.pop_back();
        }
        
    }
    
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        
        vector<vector<int>>ans;
        vector<int>ds;
        sort(nums.begin(), nums.end());
        
        func(0, nums, ds, ans);
        return ans;
    }
};

	
//Combination Sum 1
// Input: candidates = [2,3,6,7], target = 7
// Output: [[2,2,3],[7]]
// Explanation:
// 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
// 7 is a candidate, and 7 = 7.
// These are the only two combinations.	
	class Solution {
public:
    void func(int ind, int target, vector<int>&candidates, vector<vector<int>>&ans, vector<int>&ds)
    {
        if(ind==candidates.size())
        {
            if(target==0)
            {
                ans.push_back(ds);
            }
            return;
        }
        
        if(candidates[ind]<=target)
        {
            ds.push_back(candidates[ind]);
            func(ind, target-candidates[ind], candidates, ans, ds);
            ds.pop_back();
        }
        func(ind+1, target, candidates, ans, ds);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        
        vector<vector<int>>ans;
        vector<int>ds;
        
        func(0, target, candidates, ans, ds);
        return ans;
    }
};

//Combination Sum II
// Each number in candidates may only be used once in the combination.
// Input: candidates = [10,1,2,7,6,1,5], target = 8
// Output: 
// [
//	 [1,1,6],
//	 [1,2,5],
//	 [1,7]
//	 [2,6]
// ]

class Solution {
    public: 
    void findCombination(int ind, int target, vector<int> &arr, vector<vector<int>> &ans, vector<int>&ds) {
        if(target==0) {
            ans.push_back(ds);
            return;
        }        
        for(int i = ind;i<arr.size();i++) {
            if(i>ind && arr[i]==arr[i-1]) continue; 
            if(arr[i]>target) break; 
            ds.push_back(arr[i]);
            findCombination(i+1, target - arr[i], arr, ans, ds); 
            ds.pop_back(); 
        }
    }
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans; 
        vector<int> ds; 
        findCombination(0, target, candidates, ans, ds); 
        return ans; 
    }
};

	
//Palindrome Partitioning
	
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string> > res;
        vector<string> path;
        func(0, s, path, res);
        return res;
    }
    
    void func(int index, string s, vector<string> &path, 
              vector<vector<string> > &res) {
        if(index == s.size()) {
            res.push_back(path);
            return;
        }
        for(int i = index; i < s.size(); ++i) {
            if(isPalindrome(s, index, i)) {
                path.push_back(s.substr(index, i - index + 1));
                func(i+1, s, path, res);
                path.pop_back();
            }
        }
    }
    
    bool isPalindrome(string s, int start, int end) {
        while(start <= end) {
            if(s[start++] != s[end--])
                return false;
        }
        return true;
    }
};
	
//Permutation Sequence
	
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string> > res;
        vector<string> path;
        func(0, s, path, res);
        return res;
    }
    
    void func(int index, string s, vector<string> &path, 
              vector<vector<string> > &res) {
        if(index == s.size()) {
            res.push_back(path);
            return;
        }
        for(int i = index; i < s.size(); ++i) {
            if(isPalindrome(s, index, i)) {
                path.push_back(s.substr(index, i - index + 1));
                func(i+1, s, path, res);
                path.pop_back();
            }
        }
    }
    
    bool isPalindrome(string s, int start, int end) {
        while(start <= end) {
            if(s[start++] != s[end--])
                return false;
        }
        return true;
    }
};
	
//Recurssion And Backtracking
//RAT IN MAZE
	
 void solve(int i, int j, vector<vector<int>> &a, int n, vector<string> &ans, string move, 
    vector<vector<int>> &vis, int di[], int dj[]) {
        if(i==n-1 && j==n-1) {
            ans.push_back(move);
            return; 
        }
        string dir = "DLRU"; 
        for(int ind = 0; ind<4;ind++) {
            int nexti = i + di[ind]; 
            int nextj = j + dj[ind]; 
            if(nexti >= 0 && nextj >= 0 && nexti < n && nextj < n && !vis[nexti][nextj] && a[nexti][nextj] == 1) {
                vis[i][j] = 1; 
                solve(nexti, nextj, a, n, ans, move + dir[ind], vis, di, dj);
                vis[i][j] = 0; 
            }
        }
        // downward
        // if(i+1<n && !vis[i+1][j] && a[i+1][j] == 1) {
        //     vis[i][j] = 1; 
        //     solve(i+1, j, a, n, ans, move + 'D', vis);
        //     vis[i][j] = 0; 
        // }
        
        // // left
        // if(j-1>=0 && !vis[i][j-1] && a[i][j-1] == 1) {
        //     vis[i][j] = 1; 
        //     solve(i, j-1, a, n, ans, move + 'L', vis);
        //     vis[i][j] = 0; 
        // }
        
        // // right 
        // if(j+1<n && !vis[i][j+1] && a[i][j+1] == 1) {
        //     vis[i][j] = 1; 
        //     solve(i, j+1, a, n, ans, move + 'R', vis);
        //     vis[i][j] = 0; 
        // }
        
        // // upward
        // if(i-1>=0 && !vis[i-1][j] && a[i-1][j] == 1) {
        //     vis[i][j] = 1; 
        //     solve(i-1, j, a, n, ans, move + 'U', vis);
        //     vis[i][j] = 0; 
        // }
    }
    public:
    vector<string> findPath(vector<vector<int>> &m, int n) {
        vector<string> ans;
        vector<vector<int>> vis(n, vector<int> (n, 0)); 
        int di[] = {+1, 0, 0, -1}; 
        int dj[] = {0, -1, 1, 0}; 
        if(m[0][0] == 1) solve(0,0,m,n, ans, "", vis, di, dj); 
        return ans; 
    }

	
//DIVIDE AND CONQUER
	
//BINARY SEARCH
//N th root of a number {Time complexity is N * log m*10^d} where if ans need to be found upto d decimal places and n can be reduce to log n by pow function technique of even and odd
	//	     d	
	// N log(M*10)
	//	2

	
double multipy(double number, int n)  // instead of taking a for loop we can also using even odd method to take the power in O(log n)
{
    double ans=1.0;
    for(int i=1;i<=n;i++)
    {
        ans=ans*number;
    }
    return ans;
}
double getNthRoot(int n, int m)
{
    double low=1;
    double high=m;
    double eps=1e-6;
    
    while((high-low)>eps)
    {
        double mid=(low+high)/2.0;
        if(multipy(mid, n) < m)
        {
            low=mid;
        }
        else
        {
            high=mid;
        }
    }
    cout<<low<<" "<<high<<endl;
    cout<<pow(m, (double)(1.0/(double)n));
}
//Median in Rowwise and Columnwise sorted array
// first binary search to find the value in binary matrix and then another binary search for finding smaller number in each row for that particular value on that matrix

int countSmallerThanMid(vector<int> &row, int mid) 
{
    int l = 0;
    int h = row.size() - 1; 
    while(l <= h) 
    {
        int md = (l + h) >> 1; 
        if(row[md] <= mid)
	{
            l = md + 1;
        }
        else 
	{
            h = md - 1;
        }
    }
    return l; 
}
int Solution::findMedian(vector<vector<int> > &A) 
{
    int low = INT_MIN;
    int high = INT_MAX; 
    int n = A.size();
    int m = A[0].size(); 
    while(low <= high)
    {
        int mid = (low + high) >> 1; 
        int cnt = 0;
        for(int i = 0;i<n;i++)
	{
            cnt += countSmallerThanMid(A[i], mid); 
        }
        
        if(cnt <= (n * m) / 2) 
	 low = mid + 1; 
        else 
	 high = mid - 1; 
    }
    return low; 
}
	
	
//Single in array of duplicates
// as array is sorted so there will be rule that one element will be on odd index and one element will be on even index so you just need to find left half
// and next to the left half will be the ans and single number in array
// high is put over size-2 cause of last element can be the single one
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int low=0;
        int high=nums.size()-2;
        
        while(low<=high)
        {
            int mid=(low+high)/2;
            if(nums[mid]==nums[mid^1])
            {
                low=mid+1;
            }
            else
            {
                high=mid-1;
            }
        }
        return nums[low];
    }
};
	
// Search Element in rotated sorted array
//log n
class Solution {
public:
    int search(vector<int>& a, int target) {
        int low = 0, high = a.size() - 1; 
        while(low <= high) {
            int mid = (low + high) >> 1; 
            if(a[mid] == target) return mid; 
            
            // the left side is sorted
            if(a[low] <= a[mid]) 
	    {
                if(target >= a[low] && target <= a[mid]) //if your element lies on left half or not
		{
                    high = mid - 1; 
                }
                else {
                    low = mid + 1; 
                }
            }
            else //right side
	    {
                if(target >= a[mid] && target <= a[high])  // if your element lies on right half or not
		{
                    low = mid + 1; 
                }
                else 
		{
                    high = mid - 1; 
                }
            }
        } 
        return -1; 
    }
};
	
//Median of two sorted arrays
// find partition and then check even and odd size and then log(min(m, n)); 	
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) 
    {
        if(nums2.size() < nums1.size())
		return findMedianSortedArrays(nums2, nums1);
	    
        int n1 = nums1.size();
        int n2 = nums2.size(); 
        int low = 0, high = n1;
        
        while(low <= high)
	{
            int cut1 = (low+high) >> 1;
            int cut2 = (n1 + n2 + 1) / 2 - cut1; 
            
        
            int left1 = cut1 == 0 ? INT_MIN : nums1[cut1-1];
            int left2 = cut2 == 0 ? INT_MIN : nums2[cut2-1]; 
            
            int right1 = cut1 == n1 ? INT_MAX : nums1[cut1];
            int right2 = cut2 == n2 ? INT_MAX : nums2[cut2]; 
            
            
            if(left1 <= right2 && left2 <= right1)
	    {
                if( (n1 + n2) % 2 == 0 ) 
                    return (max(left1, left2) + min(right1, right2)) / 2.0; 
                else 
                    return max(left1, left2); 
            }
            else if(left1 > right2)
	    {
                high = cut1 - 1; 
            }
            else 
	    {
                low = cut1 + 1; 
            }
        }
        return 0.0; 
    }
};
	
//K th element in final sorted array
//log(min(m, n))
	
    int kthElement(int arr1[], int arr2[], int n, int m, int k)
    {
        if(n > m) 
	{
            return kthElement(arr2, arr1, m, n, k); 
        }
        
        int low = max(0,k-m);
	int high = min(k,n);
        
        while(low <= high) 
	{
            int cut1 = (low + high) >> 1; 
            int cut2 = k - cut1; 
		
            int l1 = cut1 == 0 ? INT_MIN : arr1[cut1 - 1]; 
            int l2 = cut2 == 0 ? INT_MIN : arr2[cut2 - 1];
            int r1 = cut1 == n ? INT_MAX : arr1[cut1]; 
            int r2 = cut2 == m ? INT_MAX : arr2[cut2]; 
            
            if(l1 <= r2 && l2 <= r1) 
	    {
                return max(l1, l2);
            }
            else if (l1 > r2) 
	    {
                high = cut1 - 1;
            }
            else
	    {
                low = cut1 + 1; 
            }
        }
        return 1; 
    }
    
// Minimum Allocation of Pages
int ispossible(vector<int>&arr, int currmidpagelimit, int student)
{
    int cnt=0;
    int sumpagesallocated=0;
    
    for(int i=0;i<arr.size();i++)
    {
        if(sumpagesallocated+arr[i] > currmidpagelimit)
        {
            cnt++;
            sumpagesallocated=arr[i];
            
            if(sumpagesallocated>currmidpagelimit)
            return false;
        }
        else
        {
            sumpagesallocated+arr[i];
        }
    }
    
    if(cnt<students)
    return true;
    else
    return false;
}

int books(vector<int>&arr, int k)
{
    int n=A.size();
    
    if(k>n)
    return -1;
    
    int low=arr[0];
    int high=0;
    
    for(int i=0;i<n;i++)
    {
        high=high+arr[i];
        low=min(low, A[i]);
    }
    
    int res=-1;
    while(low<=high)
    {
        int mid=(high+low)>>1;
        if(ispossible(arr, mid, k))
        {
            res=mid;
            high=mid-1;
        }
        else
        {
            low=mid+1;
        }
    }
    return low;
    
}

//Aggressive Cows
#include<bits/stdc++.h>
using namespace std;

bool ispossible(int arr[], int n, int cows, int minDist)
{
    int cntCows=1;
    int lastPlacedCows=arr[0];
    
    for(int i=1;i<n;i++)
    {
        if(arr[i]-lastPlacedCows >= minDist)
        {
            cntCows++;
            lastPlacedCows=arr[i];
        }
    }
    
    if(cntCows>=cows)
    return true;
    else
    return false;
}


int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        int n, cows;
        cin>>n>>cows;
        
        int arr[n];
        
        for(int i=0;i<n;i++)
        {
            cin>>arr[i];
        }
        sort(arr, arr+n);
        
        int low=1;
        int high=arr[n-1] - arr[0];
        
        while(low<=high)
        {
            int mid=(high+low)/2;
            
            if(ispossible(arr, n, cows, mid))
            {
                low=mid+1;
            }
            else
            {
                high=mid-1;
            }
        }
        cout<<high<<endl;
    }
    return 0;
}

//Stack Using Array

void MyStack :: push(int x)
{
    // Your Code
    arr[top++]=x;
}

//Function to remove an item from top of the stack.
int MyStack :: pop()
{
    // Your Code 
    if(top==-1)
    return -1;
    
    return arr[--top];
}
	
	
//Queue using array
	
void push(int x)
{
    if(cnt==n)
    return -1;
    
    arr[rear%n]=x;
    rear++;
    cnt++;
}

int pop()
{
    if(cnt==0)
    return -1;
    arr[front%n]=-1;
    front++;
    cnt--;
}

int top()
{
    if(cnt==0)
    return -1;
    
    return arr[front%n];
}
	

//Queue using Stack 
//push O(1)  , pop O(n) Theta(1) Amortized
stack<int>s1, s2;

void enqueue(int x)
{
    s1.push(x);
}

int pop()
{
    if(s1.empty())
    {
        if(s2.empty())
        {
            return -1;
        }
        else
        {
            while(!s1.empty())
            {
                s2.push(s1.top());
                s1.pop();
            }
        }
    }
    int res=s1.top();
    s1.pop();
    return res;
}




//Using Single stack
//push O(1), pop O(n) 
stack<int>s1;
void enqueue(int x)
{
    s1.push(x);
}

int pop()
{
    if(s1.empty())
    {
        return -1;
    }
    
    int x=s1.top();
    s1.pop();
    
    if(s1.empty())
    {
        return x;
    }
    
    int y=pop();
    push(x);
    return y;
}
	
	
//Stack using Queue
//Using Two queue int push put the new element in q2 then put all element of q1 to q2
//then swap q1, q2
//Push O(n), Pop O(1)
queue<int>q1, q2;

void push(int x)
{
    q2.push(x);
    
    while(!q1.empty())
    {
        q2.push(q1.top());
        q1.pop();
    }
    
    swap(q1, q2);
}

void pop()
{
    if(q1.empty())
    {
        return -1;
    }
    else
    {
        int x=q1.top();
        q1.pop();
        return x;
    }
}




//Stack Using single queue
queue<int>q1;

void push(int x)
{
        q1.push(x);
        int size=q1.size();
        
        for(int i=0;i<size-1;i++)
        {
            int tmp=q1.front();
            q1.pop;
            q1.push(tmp);
        }
}

void pop()
{
    if(q1.empty())
    {
        return -1;
    }
    else
    {
        int res=q1.front();
        q1.pop();
        return res;
    }
}
	
//Next Greater Element
	

#include<bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cin>>n;
    int arr[n];
    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }
    stack<int>s;
    vector<int>ans(n, -1);
    for(int i=n-1;i>=0;i--)
    {
        
        while(!s.empty() && arr[i]>s.top())
        s.pop();
        
        if(!s.empty())
        {
            ans[i]=s.top();
        }
        
        s.push(arr[i]);
    }
    for(int i=0;i<ans.size();i++)
    {
        cout<<ans[i]<<" ";
    }
    return 0;
}
	
//


