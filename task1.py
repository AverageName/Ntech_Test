def findMaxSubArray(nums):
    ans = nums[0]
    sum_ = 0
    l = 0
    l_ans = 0
    r = 0
    for i in range(len(nums)):
        sum_ += nums[i]
        
        if sum_ >= ans:
            ans = sum_
            r = i
            l_ans = l
        
        if sum_ < 0:
            sum_ = 0
            l = i + 1
    
    return nums[l_ans:r+1]
    

if __name__ == "__main__":
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    #[4, -1, 2, 1]
    print(findMaxSubArray(nums))
    nums = [-4]
    #[-4]
    print(findMaxSubArray(nums))
    nums = [-10, 1, 2, -4, 7]
    #[7]
    print(findMaxSubArray(nums))
