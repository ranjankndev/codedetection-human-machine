# Error analysis - task A (gpad, split `ood_lang`)

The 10 most *confident* mistakes (confidence after softmax, no temperature).
Confident errors are the ones that would actually harm a user, so they are the ones worth reading.

## 1. true `machine` -> predicted `human` (confidence 0.97)
- language=Java, generator=Qwen/Qwen2.5-Coder-7B
- row_id `A-665f94a40ab9a63d-338698`

```
public class Main {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n=sc.nextInt();
		int m=sc.nextInt();
		String board[][]=new String[n][m];
		for(int i=0;i<n;i++)
			for(int j=0;j<m;j++)
				board[i][j]=sc.next();
		int white=0;
		for(int i=0;i<n;i++) {
```

## 2. true `machine` -> predicted `human` (confidence 0.96)
- language=Java, generator=meta-llama/Llama-3.1-8B-Instruct
- row_id `A-79bbc14e101c5006-262900`

```
, runner should print output answer

import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.StringTokenizer;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;

```

## 3. true `machine` -> predicted `human` (confidence 0.95)
- language=Java, generator=bigcode/starcoder
- row_id `A-893e114fa2ff65a2-318844`

```
public:
 	int maxTwoSubArrays(vector<int>& nums, int firstLen, int secondLen) {
		int n = nums.size();
		if (n >= firstLen && n >= secondLen) {
			vector<int> _max(n - firstLen + 1, nums[firstLen - 1]), _min(n - secondLen + 1, nums[secondLen - 1]);
			for (int i = firstLen; i < n; ++i) {
				if (_max[i - firstLen + 1] < nums[i])
					_max[i - firstLen + 1] = nums[i];
				else
					_min[i - secondLen + 1] = std::min(_min[i - secondLen + 1], nums[i]);
			}
			return max(_max.back() + _min.back(), max(_max.front() + _min.front(), 0));
```

## 4. true `human` -> predicted `machine` (confidence 0.94)
- language=Java, generator=human
- row_id `A-71a36923e60e9c0b-476858`

```
// package c1721;

//
// Educational Codeforces Round 134 (Rated for Div. 2) 2022-08-27 07:35
// E. Prefix Function Queries
// https://codeforces.com/contest/1721/problem/E
// time limit per test 2 seconds; memory limit per test 256 megabytes
// public class Pseudo for 'Source should satisfy regex [^{}]*public\s+(final)?\s*class\s+(\w+).*'
//
// You are given a string s, consisting of lowercase Latin letters.
//
// You are asked q queries about it: given another string t, consisting of lowercase Latin letters,
```

## 5. true `human` -> predicted `machine` (confidence 0.93)
- language=Java, generator=human
- row_id `A-2afd7be341150d06-36314`

```
import java.io.*;

import java.util.*;



/**

 * A simple template for competitive programming problems.

 */

```

## 6. true `human` -> predicted `machine` (confidence 0.93)
- language=Java, generator=human
- row_id `A-db2881ed34a20bf7-442674`

```
class Solution {
    public int[] getMaximumXor(int[] nums, int maximumBit) {
        int n = nums.length;
    int[] ans = new int[n];
    int xor = 0;
    for (int v : nums) {
      xor ^= v;
    }
    // System.out.format("xor: %d\n", xor);
    int mask = (1 << maximumBit) - 1;
    for (int i = 0; i < n; i++) {
      ans[i] = mask ^ xor;
```

## 7. true `human` -> predicted `machine` (confidence 0.93)
- language=Java, generator=human
- row_id `A-d60dacea4439033d-430865`

```
import java.util.*;

import java.io.*;



public class Main {

  static StringBuilder sb;

  static dsu dsu;

```

## 8. true `machine` -> predicted `human` (confidence 0.93)
- language=Java, generator=meta-llama/Llama-3.1-8B-Instruct
- row_id `A-774894f0e920be20-446307`

```
and do not put lp in comments

import java.util.*;
import java.lang.*;
import java.io.*;

public class B {

	public static void main(String[] args) throws IOException {
		frame();
		int n = sc.nextInt();
		int[] a = new int[n];
```

## 9. true `human` -> predicted `machine` (confidence 0.93)
- language=Java, generator=human
- row_id `A-4a603ce6158130dc-262594`

```
import javax.sound.midi.SysexMessage;

import java.io.*;

import java.util.Arrays;

import java.util.Comparator;

import java.util.PriorityQueue;

import java.util.StringTokenizer;

```

## 10. true `human` -> predicted `machine` (confidence 0.93)
- language=Java, generator=human
- row_id `A-19b2ce3d0267b9a2-494023`

```
import java.io.*;

import java.util.*;



import static java.lang.Math.*;

import static java.util.Arrays.*;



```
