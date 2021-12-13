package bike.snu.ac.kr.anticancer_drug_combination_prediction_using_documents_based_features;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org._dmis.object.BioEntityExtractor;
import org._dmis.object.BioEntityInfo;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class Articles {
	public TreeSet<String> collectDocuments() {
		TreeSet<String> documentSet = new TreeSet<String>();
		
		String pubchemPath = "./data/pubchem_terms.txt";
		TreeSet<String> drugDocuments = getDocuments(pubchemPath);
		
		String cellosaurusPath = "./data/cellosaurus_terms.txt";
		TreeSet<String> cellLineDocuments = getDocuments(cellosaurusPath);
		
		for(String s : drugDocuments) {
			documentSet.add(s);
		}
		for(String s : cellLineDocuments) {
			documentSet.add(s);
		}
		
		return documentSet;
	}
	
	public void reviseDocuments(TreeSet<String> documentSet) {
		String dictionaryPath = "./data/dictionary.txt";
		Map<Integer, String> mainTerms = getMainTerms(dictionaryPath);
		
		BioEntityExtractor bee = new BioEntityExtractor(dictionaryPath);
		
		BufferedWriter bw;
		try {
			bw = new BufferedWriter(new FileWriter("./data/documents.txt", true));
			
			for(String s : documentSet) {
				String[] split = s.split("\t");
				
				String pmid = split[0];
				String abs = split[1];
				
				String reviseAbs = abs;
				
				if(abs.length() > 10 && !pmid.equals("31291753")) {
					HashSet<BioEntityInfo> entitySet = bee.extractEntities(abs);
					
					for (BioEntityInfo entity : entitySet) {
						reviseAbs = reviseAbs.replace(entity.getName(), mainTerms.get(entity.getId()));
					}
					
					if(!abs.equals(reviseAbs)) {
						bw.write(reviseAbs);
						bw.newLine();
					}
				}
			}
			bw.flush();
			bw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private Map<Integer, String> getMainTerms(String inputPath) {
		Map<Integer, String> mainTerms = new HashMap<Integer, String>();
		
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(inputPath));
			String line = "";
			while ((line = br.readLine()) != null) {
				String[] split = line.split("\t");
				int id = Integer.parseInt(split[0].substring(1, split[0].length()));
				
				if(!mainTerms.containsKey(id)) {
					mainTerms.put(id, split[1]);
				}
				
			}

			br.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mainTerms;
	}
	
	private TreeSet<String> getDocuments(String inputPath) {
		TreeSet<String> documents = new TreeSet<String>();
		
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(inputPath));
			String line = "";
			while ((line = br.readLine()) != null) {
				String[] split = line.split("\t");
				String representativeTerm = split[0];

				if (representativeTerm.contains("/")) {
					representativeTerm = representativeTerm.replace("/", "-");
				}
				TreeSet<String> ids = new TreeSet<String>();
				for (String term : split) {
					if (!term.startsWith("UNII-") && !term.startsWith("ChemBio") && !term.startsWith("JMC")) {
						TreeSet<String> extractIds = getPubmedIds(term);
						if (extractIds.size() > 0) {
							for (String id : extractIds) {
								ids.add(id);
							}
						}
					}
				}
				
				System.out.println(representativeTerm + ":" + "\t" + ids.size());

				List<String> idList = new ArrayList<String>();
				for (String id : ids) {
					idList.add(id);
				}

				TreeSet<String> idLines = new TreeSet<String>();
				String idLine = "";

				int cnt = 0;
				for (int i = 0; i < idList.size(); i++) {
					++cnt;

					idLine = idLine + idList.get(i) + ",";

					if (cnt % 200 == 0) {
						idLines.add(idLine.substring(0, idLine.length() - 1));

						idLine = "";
					}
				}

				if (idLine.length() > 0) {
					idLines.add(idLine.substring(0, idLine.length() - 1));
				}

				int cnt2 = 0;
				for (String id : idLines) {
					String getTextUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + id + "&retmode=xml";

					DocumentBuilderFactory dbFactoty = DocumentBuilderFactory.newInstance();
					DocumentBuilder dBuilder = dbFactoty.newDocumentBuilder();
					Document doc = dBuilder.parse(getTextUrl);

					doc.getDocumentElement().normalize();

					NodeList pubmedArticle = doc.getElementsByTagName("PubmedArticle");
					for (int i = 0; i < pubmedArticle.getLength(); i++) {
						Node articleNode = pubmedArticle.item(i);
						Element articleElement = (Element) articleNode;

						String pmid = "";
						NodeList idNodeList = articleElement.getElementsByTagName("PMID");
						for (int j = 0; j < idNodeList.getLength(); j++) {
							String text = idNodeList.item(j).getTextContent();
							pmid = pmid + text + " ";

						}
						pmid = pmid.trim();

						String abs = "";
						NodeList absList = articleElement.getElementsByTagName("Abstract");
						for (int j = 0; j < absList.getLength(); j++) {
							String text = absList.item(j).getTextContent();
							abs = abs + text + " ";
						}

						abs = abs.trim();

						if (pmid.length() > 0 && abs.length() > 10) {
							documents.add(pmid + "\t" + abs);
						}
					}
					
					++cnt2;

					if (cnt2 % 5 == 0) {
						System.out.println(cnt2 * 200);
					}
				}
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return documents;
	}
	
	private TreeSet<String> getPubmedIds(String keyword) {
		TreeSet<String> ids = new TreeSet<String>();

		try {
			if(keyword.length() > 2 && !keyword.contains("|")) {
				if(keyword.contains("&")) {
					keyword = keyword.replaceAll("&", "");
				} else if(keyword.contains(";")) {
					keyword = keyword.replaceAll(";", " ");
				}
				
				if(keyword.contains(" ")) {
					keyword = keyword.replaceAll(" ", "%20");
				} else {
					keyword = URLEncoder.encode(keyword, "US-ASCII");
				}
				
				String url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=" + keyword;

				DocumentBuilderFactory dbFactoty = DocumentBuilderFactory.newInstance();
				DocumentBuilder dBuilder = dbFactoty.newDocumentBuilder();
				
				Document doc = dBuilder.parse(url);
				
				doc.getDocumentElement().normalize();
				
				NodeList eSearchResult = doc.getElementsByTagName("eSearchResult");
				
				Node eSearchResultNode = eSearchResult.item(0);
				Element eSearchResultElement = (Element) eSearchResultNode;
				
				if(eSearchResultElement.getElementsByTagName("ErrorList").getLength() == 0) {
					if((eSearchResultElement.getElementsByTagName("WarningList").getLength() == 0) || (eSearchResultElement.getElementsByTagName("WarningList").getLength() > 0 && !keyword.contains(eSearchResultElement.getElementsByTagName("WarningList").item(0).getTextContent()))) {
						int count = Integer.parseInt(eSearchResultElement.getElementsByTagName("Count").item(0).getTextContent());
						int retMax = Integer.parseInt(eSearchResultElement.getElementsByTagName("RetMax").item(0).getTextContent());
						
						if (count > retMax) {
							url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=" + keyword + "&retmax=" + count;

							doc = dBuilder.parse(url);

							doc.getDocumentElement().normalize();

							eSearchResult = doc.getElementsByTagName("eSearchResult");
							eSearchResultNode = eSearchResult.item(0);
							eSearchResultElement = (Element) eSearchResultNode;

							NodeList idNodeList = eSearchResultElement.getElementsByTagName("Id");
							for (int i = 0; i < idNodeList.getLength(); i++) {
								String id = idNodeList.item(i).getTextContent();
								ids.add(id);
							}

						} else {
							NodeList idNodeList = eSearchResultElement.getElementsByTagName("Id");
							for (int i = 0; i < idNodeList.getLength(); i++) {
								String id = idNodeList.item(i).getTextContent();
								ids.add(id);
							}
						}
					}
				}
			}
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return ids;
	}
}
