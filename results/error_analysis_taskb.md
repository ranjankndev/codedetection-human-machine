# Error analysis - task B (gpad_both, split `ood_gen`)

The 10 most *confident* mistakes (confidence after softmax, no temperature).
Confident errors are the ones that would actually harm a user, so they are the ones worth reading.

## 1. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-ad877520d774e447-532`

```
class S{int m(int[] q){int b=q[0];for(int x:q)if(x>b)b=x;return b;}}
// case 802181559-225643145
```

## 2. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-9031bf3f8c011740-178`

```
class S{int m(int[] v){int b=v[0];for(int x:v)if(x>b)b=x;return b;}}
// case 74438556-570188916
```

## 3. true `human` -> predicted `mistral` (confidence 0.16)
- language=Python, generator=Human
- row_id `B-8b073bd9f7c34514-255`

```
def f(q):
  m=q[0]
  for x in q[1:]:
   if x>m: m=x
  return m
# case 650971990-741672676
```

## 4. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-b0361a7e7e2efd56-589`

```
class S{int m(int[] a){int b=a[0];for(int x:a)if(x>b)b=x;return b;}}
// case 485580077-444914871
```

## 5. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-6e6a12ca0ab939b9-289`

```
class S{int m(int[] xs){int b=xs[0];for(int x:xs)if(x>b)b=x;return b;}}
// case 640662119-786485284
```

## 6. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-eeff67ba519ac61f-91`

```
class S{int m(int[] xs){int b=xs[0];for(int x:xs)if(x>b)b=x;return b;}}
// case 65563152-245066387
```

## 7. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-3cb37a4df2943d9b-370`

```
class S{int m(int[] xs){int b=xs[0];for(int x:xs)if(x>b)b=x;return b;}}
// case 993629312-556160222
```

## 8. true `human` -> predicted `mistral` (confidence 0.16)
- language=Python, generator=Human
- row_id `B-47658dbf7374efe8-222`

```
def f(nums):
  m=nums[0]
  for x in nums[1:]:
   if x>m: m=x
  return m
# case 35325131-960420293
```

## 9. true `human` -> predicted `qwen` (confidence 0.16)
- language=Java, generator=Human
- row_id `B-4c29b77fbbb5dfe0-550`

```
class S{int m(int[] nums){int b=nums[0];for(int x:nums)if(x>b)b=x;return b;}}
// case 765362725-329236299
```

## 10. true `bigcode` -> predicted `mistral` (confidence 0.16)
- language=Java, generator=bigcode/starcoder2-3b
- row_id `B-510ac795b54515dd-82`

```
public class Solution {
    /**
     * computeSum over the given array.
     */
    public int computeSum(int[] numbers) {
        int result = numbers[0];
        for (int number : numbers) {
            if (number > result) {
                result = number;
            }
        }
        return result;
```
