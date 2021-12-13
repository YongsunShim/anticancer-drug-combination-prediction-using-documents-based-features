package bike.snu.ac.kr.anticancer_drug_combination_prediction_using_documents_based_features;

public class Main {

	public static void main(String[] args) {
		Articles a = new Articles();
		a.collectDocuments();
		a.reviseDocuments(a.collectDocuments());
	}

}
