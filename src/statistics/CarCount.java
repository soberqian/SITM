package statistics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import util.FileUtil;
/* 
 * ���ͳ��ִ���ͳ��
 * word count
 * 
 * @author: Qian Yang
 * */
public class CarCount {

	public static void main(String[] args) {
		List<Entry<String, Integer>> list = wordcountprocess("data/clickdata/clickutf8", "utf-8");
		//���top 50 ����Ƶ��
		int number = 0;
		for (Map.Entry<String, Integer> mapping : list) { 
			number++;
			if (number == 15) {
				break;
			}
			System.out.println(mapping.getKey() + ":" + mapping.getValue());  
		}  
	}
	public static List<Entry<String, Integer>> wordcountprocess(String file, String code)  {
		Hashtable<String, Integer> wordCount = new Hashtable<>();
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(file, docLines,code);
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line, words,"--");
			for (int i = 0; i < words.size(); i++) {
				if (!wordCount.containsKey(words.get(i))) {
					wordCount.put(words.get(i), 1);
				}else {
					wordCount.put(words.get(i), wordCount.get(words.get(i)) + 1);
				}
			}
		}
		
		//��map.entrySet()ת����list  
		List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(wordCount.entrySet());  
		Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {  
			//��������  
			public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2) {  
				//return o1.getValue().compareTo(o2.getValue());  
				return o2.getValue().compareTo(o1.getValue());  
			}  
		});  
		return list;  //�������
	}
}
