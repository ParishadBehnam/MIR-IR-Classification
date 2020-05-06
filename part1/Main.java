import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.fa.PersianAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

/**
 * Created by parishad on 12/24/17.
 */
public class Main {

    private static HashMap<String, Integer> tokens_id;
    private static String[] id_tokens;

    public static void main(String[] args) throws IOException, ParseException {
        Scanner scanner = new Scanner(System.in);
        Index indexer = new Index("persianIndex");
        Analyzer analyzer = indexer.index();

        String inp = new String(Files.readAllBytes(Paths.get("query.txt")), StandardCharsets.UTF_8);
        System.out.println("how to search? W2V / normal");
        String method = scanner.nextLine();
        if (method.equals("W2V")) {
            double[][] arrs = loadW2V();
            inp = queryExpand(inp, arrs);
        }

        System.out.println("query:\n" + inp + "\n");
        Search searcher = new Search();
        searcher.search(analyzer, inp);
    }

    private static String findBestWord(String w, double[][] arrs) {

        if (tokens_id.containsKey(w)) {
            double min = Double.MAX_VALUE;
            int minId = 0, id = tokens_id.get(w);
            double[] w_features = arrs[id];
            for (int i = 0; i < arrs.length; i++) {
                double dif = 0;
                if (i != id) {
                    //min square error method!
                    for (int j = 0; j < arrs[i].length; j++)
                        dif += Math.pow((arrs[i][j] - w_features[j]), 2.0);
                    if (dif < min) {
                        minId = i;
                        min = dif;
                    }
                }
            }

            return id_tokens[minId];
        } else return "";

    }

    private static String queryExpand(String q, double[][] arrs) {
        String output = "";
        String[] queryTokens = q.split(" ");
        String best;
        for (String w : queryTokens) {
            best = findBestWord(w, arrs);
            // expanding the query
            if (!best.equals(""))
                output += " " + best;
        }

        return q + output;
    }

    private static double[][] loadW2V() throws IOException {

        tokens_id = new HashMap<>();    //maps token to an id (helps us to access to a specific row in double[][] 2D array.
        List<String> lines = Files.readAllLines(Paths.get("persian_word2vec"));

        int n = Integer.parseInt(lines.get(0).split(" ")[0]);   //number of words
        int f = Integer.parseInt(lines.get(0).split(" ")[1]);   //number of features
        id_tokens = new String[n];  //maps id to a String in fact. used for returning a String in findBestWord.
        double[][] output = new double[n][f];

        String[] row;
        for (int i = 0; i < n; i++) {
            row = lines.get(i + 1).split(" ");
            tokens_id.put(row[0], i);
            id_tokens[i] = row[0];

            for (int j = 0; j < f; j++)
                output[i][j] = Double.parseDouble(row[j + 1]);
        }
        
        return output;
    }
}


class Index {

    private IndexWriter writer;
    public Analyzer analyzer;

    public Index(String address) throws IOException {
        Directory indexDir = FSDirectory.open(Paths.get(address));
        analyzer = new PersianAnalyzer(PersianAnalyzer.getDefaultStopSet());
        IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        writer = new IndexWriter(indexDir, cfg);
    }

    public Document makeDoc(Path address) throws IOException{
        List<String> lines = Files.readAllLines(address);
        String date = "", title = "", category = "", text = "";
        boolean s = false;
        for (int i = 0; i < lines.size(); i++) {
            if (s)
                text += lines.get(i) + " ";
            if (lines.get(i).startsWith("date")) {
                date = lines.get(i + 1);
                i++;
            }
            if (lines.get(i).startsWith("title")) {
                title = lines.get(i + 1);
                i++;
            }
            if (lines.get(i).startsWith("category")) {
                category = lines.get(i + 1);
                if (category.charAt(category.length() - 1) == ' ')
                    category = category.substring(0, category.length() - 1);
                i++;
            }
            if (lines.get(i).startsWith("text"))
                s = true;   //to be continued:D
        }

        Document doc = new Document();
        doc.add(new TextField("date", date.trim(), Field.Store.YES));
        doc.add(new TextField("title", title.trim(), Field.Store.YES));
        doc.add(new StringField("category", category.trim(), Field.Store.YES)); //StringField because doesn't need to be tokenized!
        doc.add(new TextField("text", text.trim(), Field.Store.YES));
        return doc;
    }

    public Analyzer index() throws IOException {
        Stream<Path> filePathStream=Files.walk(Paths.get("hamshahri"));
        filePathStream.forEach(filePath -> {
            Document doc = null;
            try {
                if (Files.isRegularFile(filePath)) {    //ignore Train and Test folder paths.
//                    System.out.println(filePath);
                    doc = makeDoc(filePath);
                    writer.addDocument(doc);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        writer.close();
        return analyzer;
    }

}

class Search {

    private IndexSearcher searcher;
    public Search() throws IOException {
        IndexReader indexReader = DirectoryReader.open(FSDirectory.open(Paths.get("persianIndex")));

        searcher = new IndexSearcher(indexReader);
    }

    private String[] getFeature(String raw) {

        String[] out = new String[4];
        for (int i = 0; i < 4; i++) {
            out[i] = "";    //{."date", ."title", ."category", ."text"} fields of query
        }
        String[] lines = raw.split("\n");
        boolean s = false;
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith("date")) {
                out[0] = lines[i + 1].trim();   //DATE
                s = false;
            }
            else if (lines[i].startsWith("title")) {
                out[1] = lines[i + 1].trim();   //TITLE
                s = false;
            }
            else if (lines[i].startsWith("text")) {
                //TEXT
                s = true;
            }
            else if (lines[i].startsWith("category")) {
                out[2] = lines[i + 1].trim();   //CATEGORY
                s = false;
            } else if (s)
                out[3] += lines[i];
        }

        return out;
    }

    private String makeQuery(String raw) {

        String[] features = getFeature(raw);
        String[] names = {"date", "title", "category"};

        String s = "";
        s += "text: (" + features[3] + ")";     // making query as BooleanQuery does.
        String[] tokens;
        for (int i = 0; i < 3; i++) {
            if (!features[i].equals("")) {
                tokens = features[i].split(" ");
                for (String token : tokens) {
                    s += " AND " + names[i] + ": (" + token + ")";
                }
            }
        }

        return s;
    }

    public void search(Analyzer analyzer, String q) throws ParseException, IOException {
        Query query = new QueryParser("text", analyzer).parse(makeQuery(q));
//        System.out.println(makeQuery(q));
//        System.out.println(query);
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
            Document doc = searcher.doc(hits.scoreDocs[i].doc);
            System.out.println("category: " + doc.get("category"));
            System.out.println("date: " + doc.get("date"));
            System.out.println("title: " + doc.get("title"));
            System.out.println("text:\n" + doc.get("text") + "\n");
        }

    }
}