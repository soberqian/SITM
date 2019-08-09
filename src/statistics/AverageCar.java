package statistics;

import java.util.ArrayList;

import util.FileUtil;

public class AverageCar {

	public static void main(String[] args) {
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines("data/clickdata/clickutf8", docLines,"utf-8");
		double count = 0.0;
		for (int i = 0; i < docLines.size(); i++) {
//			System.out.println(docLines.get(i));
			count += docLines.get(i).split("--").length;
		}
		System.out.println(count/docLines.size());
	}

}
