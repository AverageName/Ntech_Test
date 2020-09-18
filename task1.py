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
