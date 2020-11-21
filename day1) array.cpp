cate//Find the duplicate in an array of N+1 integers. 
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
