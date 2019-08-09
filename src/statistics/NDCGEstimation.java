package statistics;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import util.FileUtil;
import util.NDCG;

public class NDCGEstimation {
	private static int topK = 1;
	public static void main(String[] args) throws IOException {
		//排序原始数据
		List<Entry<String, Integer>> list = CarCount.wordcountprocess("data/clickdata/clickutf8", "utf-8");
		//输出top K 个高频词
		List<String> realData = new ArrayList<>();
		int number = 0;
		for (Map.Entry<String, Integer> mapping : list) { 
			number++;
			if (number > topK) {
				break;
			}
			realData.add(mapping.getKey());
		} 
		//读取影响力词分布的结果
		List<String> predictionData = new ArrayList<>();
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines("data/clickdata/SparseBLDA_backgroundtopic_word_70.txt", docLines,"utf-8");
		for (int i = 1; i < docLines.size(); i++) {//从第二行开始读取
			if (i > topK) {
				break;
			}
			predictionData.add(docLines.get(i).split(":")[0].trim());
		}
		System.out.println(realData.size() + "\t" + predictionData.size());
		//计算NDCG
		double ndcgValue = NDCG.calculateNDCG(realData, predictionData);
		System.out.println(ndcgValue);
	}

	
}
