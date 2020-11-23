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
           
