package ubicomp.falldetector2;

import android.content.BroadcastReceiver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.os.Environment;
import android.os.Handler;
import android.provider.BaseColumns;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.aware.Accelerometer;
import com.aware.Aware;
import com.aware.Aware_Preferences;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "FallDetector2";

    // every 60ms
    public static int SAMPLING_RATE = 60000;

    // threshold for acceleration magnitude (sqrt(a^2+b^2+c^2)), including earth gravity g
    public static int LOWER_THRESHOLD = 4;
    // fft array length
    public static int WINDOW_SIZE = 16;
    // size of the analysed time window
    public static int WINDOW_TIME_THRESHOLD = (SAMPLING_RATE/1000)*WINDOW_SIZE;

    private static Instances training_data, training_data_normal;

    FallReceiver rec;

    private static J48 FFTrf;
    private static J48 NormalRF;

    private static Context this_context;

    Button false_event_button,drop_button;
    ToggleButton false_monitoring_button, analyse_button;

    public static LinkedList<Double> latestValues;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this_context = getApplicationContext();
        setContentView(R.layout.activity_main);

        false_event_button = (Button) findViewById(R.id.false_event_button);
        drop_button = (Button) findViewById(R.id.drop_button);
        false_monitoring_button = (ToggleButton) findViewById(R.id.false_button);
        analyse_button = (ToggleButton) findViewById(R.id.analyse_button);

        enableButtons(false);

        Intent aware = new Intent(this, Aware.class);
        startService(aware);

        Aware.setSetting(this, Aware_Preferences.FREQUENCY_ACCELEROMETER, SAMPLING_RATE);
        Aware.startAccelerometer(this);

        rec = new FallReceiver();
        IntentFilter filt = new IntentFilter();
        filt.addAction(Accelerometer.ACTION_AWARE_ACCELEROMETER);
        registerReceiver(rec, filt);

        setButtonListeners();

    }

    private void arffDump() {
        // store training data in arff

        File path = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_MOVIES);
        ArffSaver saver = new ArffSaver();
        try {
            saver.setInstances(training_data);
            saver.setFile(new File(path, "fft"+WINDOW_SIZE+".arff"));
            //saver.writeBatch();

            saver.setInstances(training_data_normal);
            saver.setFile(new File(path, "normal"+WINDOW_SIZE+".arff"));
            saver.writeBatch();
            Log.d(TAG, "dir: " + path.getAbsolutePath());
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setButtonListeners() {

        drop_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                enableButtons(false);
                Toast.makeText(this_context, "Storing drop event", Toast.LENGTH_SHORT).show();
                TrainingDataHelper helper = new TrainingDataHelper(this_context);
                SQLiteDatabase db = helper.getWritableDatabase();
                ContentValues cVals = new ContentValues();
                String values = "";
                for (int i = 0; i < latestValues.size(); i++) {
                    values += latestValues.get(i) + ",";
                }
                cVals.put(TrainingStorage.TrainingEntry.VALUES, values);
                cVals.put(TrainingStorage.TrainingEntry.LABEL, "drop");
                db.insert(TrainingStorage.TrainingEntry.TABLE_NAME, null, cVals);
                db.close();
                helper.close();
                Log.d(TAG, "inserted: " + values);
                false_triggered = false;
            }
        });
        false_event_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                enableButtons(false);
                Toast.makeText(this_context, "Storing false event", Toast.LENGTH_SHORT).show();
                TrainingDataHelper helper = new TrainingDataHelper(this_context);
                SQLiteDatabase db = helper.getWritableDatabase();
                ContentValues cVals = new ContentValues();
                String values = "";
                for (int i = 0; i < latestValues.size(); i++) {
                    values += latestValues.get(i) + ",";
                }
                cVals.put(TrainingStorage.TrainingEntry.VALUES, values);
                cVals.put(TrainingStorage.TrainingEntry.LABEL, "false");
                db.insert(TrainingStorage.TrainingEntry.TABLE_NAME, null, cVals);
                db.close();
                helper.close();
                Log.d(TAG, "inserted: " + values);
                false_triggered = false;
            }
        });

        analyse_button.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (FFTrf == null) {
                    new Handler().post(new Runnable() {
                        @Override
                        public void run() {
                            buildClassifiers();
                        }
                    });
                }
                drop_button.setEnabled(!isChecked);
                false_event_button.setEnabled(!isChecked);
                false_monitoring_button.setEnabled(!isChecked);
            }
        });

    }

    private void buildClassifiers() {
        /*
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(getResources().getAssets().open("training_data.arff")));
            training_data = new Instances(reader);
            training_data.setClassIndex(training_data.numAttributes() - 1);

        } catch (Exception e) {
            e.printStackTrace();
        }
        */

        String[] options = new String[1];
        options[0] = "-U";
        ArrayList<Attribute> atts = new ArrayList<Attribute>(WINDOW_SIZE+2);
        ArrayList<Attribute> normal_atts = new ArrayList<Attribute>(WINDOW_SIZE+3);

        atts.add(new Attribute("max_freq"));
        normal_atts.add(new Attribute("max_accel"));
        normal_atts.add(new Attribute("min_accel"));
        int i = 1;
        while (i < WINDOW_SIZE+1) {
            atts.add(new Attribute("coeff"+i));
            normal_atts.add(new Attribute("accel"+i));
            i++;
        }
        List<String> labelVals = new ArrayList<>();
        labelVals.add("drop");
        labelVals.add("false");
        Attribute labelAttr = new Attribute("label", labelVals);
        atts.add(labelAttr);
        normal_atts.add(labelAttr);

        training_data = new Instances("Event", atts, 0);
        training_data.setClassIndex(training_data.numAttributes()-1);

        training_data_normal = new Instances("Event", normal_atts, 0);
        training_data_normal.setClassIndex(training_data_normal.numAttributes()-1);

        FFTrf = new J48();
        NormalRF = new J48();
        try {
            FFTrf.setOptions(options);
            NormalRF.setOptions(options);
        } catch (Exception e) {
        }

        TrainingDataHelper helper = new TrainingDataHelper(this_context);
        SQLiteDatabase db = helper.getReadableDatabase();

        // reset database
        //helper.onUpgrade(db, 2, 5);

        Cursor c = db.query(TrainingStorage.TrainingEntry.TABLE_NAME,
                null,null,null,null,null,null);

        FFT fft = new FFT(WINDOW_SIZE);
        ArrayList<TrainingTuple> trainingData = new ArrayList<>();
        String rowVal, label;
        String[] rowVals;
        ArrayList<Double> rowNumVals;
        c.moveToFirst();
        for (i = 0; i < c.getCount(); i++) {
            label = c.getString(2);
            rowVal = c.getString(1);
            rowVals = rowVal.split(",");
            rowNumVals = new ArrayList<>();
            for (int x = 0; x < (rowVals.length-1); x++) {
                rowNumVals.add(Double.valueOf(rowVals[x]));
            }
            trainingData.add(new TrainingTuple(rowNumVals, label));
            //Log.d(TAG, "Added values with label " + label);
            c.moveToNext();
        }
        c.close();

        for (TrainingTuple val : trainingData) {
            double[] real = new double[WINDOW_SIZE];

            i = 0;
            while (i < val.vals.size()) {
                real[i] = val.vals.get(i);
                i++;
            }
            //Complex[] result = fft.transform(values, TransformType.FORWARD);
            double[] imag = new double[WINDOW_SIZE];
            i = 0;
            while (i < WINDOW_SIZE){
                imag[i] = 0;
                i++;
            }
            fft.fft(real, imag);

            double[] row = new double[real.length+2];
            double[] normal_row = new double[real.length+3];

            row[0] = 0;
            normal_row[0] = 0;
            normal_row[1] = 50;
            if (val.label.equals("false")) {
                row[training_data.classIndex()] = 1;
                normal_row[training_data_normal.classIndex()] = 1;
            }
            else {
                row[training_data.classIndex()] = 0;
                normal_row[training_data_normal.classIndex()] = 0;
            }

            i = 1;
            while (i < real.length+1) {
                double value = Math.sqrt(real[i-1]*real[i-1] + imag[i-1]*imag[i-1]);
                if (value > row[0]) row[0] = value;
                row[i] = value;
                i++;
            }
            i = 1;
            while (i < val.vals.size()+1) {
                double value = val.vals.get(i-1);
                if (value > normal_row[0]) normal_row[0] = value;
                if (value < normal_row[1]) normal_row[1] = value;
                normal_row[i] = value;
                i++;
            }
            training_data.add(new OmaInstance(1.0, row));
            training_data_normal.add(new OmaInstance(1.0, normal_row));
        }

        try {
            NormalRF.buildClassifier(training_data_normal);
            FFTrf.buildClassifier(training_data);
        } catch (Exception e) {
            Log.d(TAG, "Error building classifier");
            e.printStackTrace();
        }
        crossValidate();
        arffDump();
    }

    private void crossValidate() {
        Log.d(TAG, "Cross validating");
        try {
            Evaluation eval = new Evaluation(training_data_normal);
            eval.crossValidateModel(NormalRF, training_data_normal, 10, new Random(1));
            Log.d(TAG, "Finished NORMAL rf: " + eval.toSummaryString());

            eval = new Evaluation(training_data);
            eval.crossValidateModel(FFTrf, training_data, 10, new Random(1));
            Log.d(TAG, "Finished FFT rf: " + eval.toSummaryString());
        }
        catch (Exception e) {
            Log.d(TAG, "failed to crossvalidate");
            e.printStackTrace();
        }
    }

    private void enableButtons(boolean enabled) {
        false_event_button.setEnabled(enabled);
        drop_button.setEnabled(enabled);
        false_monitoring_button.setEnabled(!enabled);
    }

    private boolean detectingFalse() {
        return (false_monitoring_button.isChecked());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        Aware.stopAccelerometer(this);

        /*Intent aware = new Intent(this, Aware.class);
        stopService(aware);
        */
        unregisterReceiver(rec);
    }

    public static void classify(Instance fft, Instance normal) {
        try {
            double result = FFTrf.classifyInstance(fft);
            double normal_result = NormalRF.classifyInstance(normal);
            Log.d(TAG, "FFT: " + training_data.classAttribute().value((int) result));
            Log.d(TAG, "Normal: " + training_data_normal.classAttribute().value((int) normal_result));

            Toast.makeText(this_context, training_data.classAttribute().value((int) result) + "/" + training_data_normal.classAttribute().value((int) normal_result), Toast.LENGTH_SHORT).show();
        }
        catch (Exception e) {
            Log.d(TAG, "could not classify");
            e.printStackTrace();
        }
    }

    public static boolean false_triggered = false;

    private class FallReceiver extends BroadcastReceiver {
        private final static String LOGTAG = "FallReceiver";

        private boolean ANALYSING = false;

        private LinkedList<Double> acc_values = new LinkedList<>();

        long trigger_time = 0;

        @Override
        public void onReceive(final Context context, Intent intent) {
            if (intent.hasExtra(Accelerometer.EXTRA_DATA)) {
                ContentValues c = (ContentValues) intent.getExtras().get(Accelerometer.EXTRA_DATA);
                try {
                    // if not already waiting for input
                    double a = Math.sqrt(c.getAsDouble("double_values_0") * c.getAsDouble("double_values_0") + c.getAsDouble("double_values_1") * c.getAsDouble("double_values_1") + c.getAsDouble("double_values_2") * c.getAsDouble("double_values_2"));
                    if (false_triggered && (trigger_time > (System.currentTimeMillis() - WINDOW_TIME_THRESHOLD * 3/4))) {
                        if (acc_values.size() >= WINDOW_SIZE) acc_values.removeFirst();
                        acc_values.add(a);
                        Log.d(TAG, "added " + a + " to values array during trigger");
                    }
                    else if (!false_triggered) {
                        if (acc_values.size() >= WINDOW_SIZE) acc_values.removeFirst();
                        acc_values.add(a);
                    }
                    if (a < LOWER_THRESHOLD && !false_triggered) {
                        trigger_time = System.currentTimeMillis();
                        false_triggered = true;
                        Log.d(TAG, "below threshold");
                        if (!ANALYSING) {
                            ANALYSING = true;
                            new Handler().postDelayed(new Runnable() {
                                @Override
                                public void run() {
                                    Log.d(TAG, "Triggered!");
                                    if (acc_values.size() == WINDOW_SIZE) {
                                        if (!analyse_button.isChecked()) {
                                            categorise(context, acc_values);
                                        } else {
                                            Log.d(TAG, "analysing!");
                                            analyse(context);
                                            false_triggered = false;
                                        }
                                        ANALYSING = false;
                                    }
                                    trigger_time = 0;
                                }
                            }, WINDOW_TIME_THRESHOLD * 3/4);
                        }
                    }

                } catch (NullPointerException e) {}
            }
        };

        private void categorise(Context context, final LinkedList<Double> vals) {
            final LinkedList<Double> v = vals;
            latestValues = v;
            // if detecting false, categorise this as false
            if (detectingFalse()) {
                Toast.makeText(context, "Storing negative event", Toast.LENGTH_SHORT).show();
                TrainingDataHelper helper = new TrainingDataHelper(context);
                SQLiteDatabase db = helper.getWritableDatabase();
                ContentValues cVals = new ContentValues();
                String values = "";
                for (int i = 0; i < latestValues.size(); i++) {
                    values += latestValues.get(i) + ",";
                }
                cVals.put(TrainingStorage.TrainingEntry.VALUES, values);
                cVals.put(TrainingStorage.TrainingEntry.LABEL, "false");
                db.insert(TrainingStorage.TrainingEntry.TABLE_NAME, null, cVals);
                db.close();
                helper.close();
                false_triggered = false;
                Log.d(TAG, "inserted: " + values);
            }
            // else allow the values to be stored with a label
            else {
                enableButtons(true);
                Toast.makeText(context, "Categorise the event", Toast.LENGTH_SHORT).show();
            }
        }

        private void analyse(Context context) {
            ArrayList<Attribute> atts = new ArrayList<Attribute>(WINDOW_SIZE+2);
            ArrayList<Attribute> normal_atts = new ArrayList<Attribute>(WINDOW_SIZE+3);

            atts.add(new Attribute("max_freq"));
            normal_atts.add(new Attribute("max_accel"));
            normal_atts.add(new Attribute("min_accel"));

            int i = 1;
            while (i < WINDOW_SIZE+1) {
                atts.add(new Attribute("coeff"+i));
                normal_atts.add(new Attribute("accel"+i));
                i++;
            }
            atts.add(new Attribute("label"));
            normal_atts.add(new Attribute("label"));
            try {
                FFT fft = new FFT(WINDOW_SIZE);
                /*FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
                */
                double[] real = new double[WINDOW_SIZE];

                i = 0;
                while (i < acc_values.size()) {
                    real[i] = acc_values.get(i);
                    i++;
                }
                //Complex[] result = fft.transform(values, TransformType.FORWARD);
                double[] imag = new double[WINDOW_SIZE];
                i = 0;
                while (i < WINDOW_SIZE){
                    imag[i] = 0;
                    i++;
                }
                fft.fft(real, imag);

                double[] values = new double[WINDOW_SIZE+2];
                double[] normal_values = new double[WINDOW_SIZE+3];

                values[0] = 0;
                values[training_data.classIndex()] = 0;
                normal_values[0] = 0;
                normal_values[1] = 50;
                normal_values[training_data_normal.classIndex()] = -1;
                i = 1;
                while (i < WINDOW_SIZE+1) {
                    double value = Math.sqrt(real[i-1]*real[i-1] + imag[i-1]*imag[i-1]);
                    if (value > values[0]) values[0] = value;
                    if (value > normal_values[0]) normal_values[0] = value;
                    if (value < normal_values[1]) normal_values[1] = value;

                    values[i] = value;
                    normal_values[i] = value;

                    i++;
                }

                Instances container = new Instances("Event", atts, 0);
                Instances container2 = new Instances("Event2", normal_atts, 0);
                container.add(new DenseInstance(1.0, values));
                container.setClassIndex(training_data.classIndex());
                container2.add(new DenseInstance(1.0, normal_values));
                container2.setClassIndex(training_data_normal.classIndex());
                classify(container.firstInstance(), container2.firstInstance());

            } catch (Exception e) {
                ANALYSING = false;
                e.printStackTrace();
            }
            ANALYSING = false;
        }
    }

    public final class TrainingStorage {

        // To prevent someone from accidentally instantiating the contract class,
        // give it an empty constructor.
        public TrainingStorage() {}

        /* Inner class that defines the table contents */
        public abstract class TrainingEntry implements BaseColumns {
            public static final String TABLE_NAME = "TrainingEntry";
            public static final String _ID = "_ID";
            public static final String VALUES = "value_array";
            public static final String LABEL = "label";
        }
    }

    public class TrainingDataHelper extends SQLiteOpenHelper {
        private static final String TEXT_TYPE = " TEXT";
        private static final String NUM_TYPE = " INTEGER";
        private static final String COMMA_SEP = ",";
        private static final String SQL_CREATE_ENTRIES =
                "CREATE TABLE " + TrainingStorage.TrainingEntry.TABLE_NAME + " (" +
                        TrainingStorage.TrainingEntry._ID + NUM_TYPE + " PRIMARY KEY," +
                        TrainingStorage.TrainingEntry.VALUES + TEXT_TYPE + COMMA_SEP +
                        TrainingStorage.TrainingEntry.LABEL + TEXT_TYPE +
                        ")";

        private static final String SQL_DELETE_ENTRIES =
                "DROP TABLE IF EXISTS " + TrainingStorage.TrainingEntry.TABLE_NAME;
        // If you change the database schema, you must increment the database version.
        public static final int DATABASE_VERSION = 1;
        public static final String DATABASE_NAME = "FeedReader.db";

        public TrainingDataHelper(Context context) {
            super(context, DATABASE_NAME, null, DATABASE_VERSION);
        }
        public void onCreate(SQLiteDatabase db) {
            db.execSQL(SQL_CREATE_ENTRIES);
        }
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            // This database is only a cache for online data, so its upgrade policy is
            // to simply to discard the data and start over
            db.execSQL(SQL_DELETE_ENTRIES);
            onCreate(db);
        }
        public void onDowngrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            onUpgrade(db, oldVersion, newVersion);
        }
    }

    public class FFT {

        int n, m;

        // Lookup tables.  Only need to recompute when size of FFT changes.
        double[] cos;
        double[] sin;

        double[] window;

        public FFT(int n) {
            this.n = n;
            this.m = (int) (Math.log(n) / Math.log(2));

            // Make sure n is a power of 2
            if (n != (1 << m))
                throw new RuntimeException("FFT length must be power of 2");

            // precompute tables
            cos = new double[n / 2];
            sin = new double[n / 2];
            for (int i = 0; i < n / 2; i++) {
                cos[i] = Math.cos(-2 * Math.PI * i / n);
                sin[i] = Math.sin(-2 * Math.PI * i / n);
            }
            makeWindow();
        }

        protected void makeWindow() {
            // Make a blackman window:
            // w(n)=0.42-0.5cos{(2*PI*n)/(N-1)}+0.08cos{(4*PI*n)/(N-1)};
            window = new double[n];
            for (int i = 0; i < window.length; i++)
                window[i] = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (n - 1));
        }

        public double[] getWindow() {
            return window;
        }

        /*
         00089   * fft.c
         00090   * Douglas L. Jones
         00091   * University of Illinois at Urbana-Champaign
         00092   * January 19, 1992
         00093   * http://cnx.rice.edu/content/m12016/latest/
         00094   *
         00095   *   fft: in-place radix-2 DIT DFT of a complex input
         00096   *
         00097   *   input:
         00098   * n: length of FFT: must be a power of two
         00099   * m: n = 2**m
         00100   *   input/output
         00101   * x: double array of length n with real part of data
         00102   * y: double array of length n with imag part of data
         00103   *
         00104   *   Permission to copy and use this program is granted
         00105   *   as long as this header is included.
         00106   ****************************************************************/
        public void fft(double[] x, double[] y) {
            int i, j, k, n1, n2, a;
            double c, s, e, t1, t2;
            j = 0;
            n2 = n / 2;
            for (i = 1; i < n - 1; i++) {
                n1 = n2;
                while (j >= n1) {
                    j = j - n1;
                    n1 = n1 / 2;
                }
                j = j + n1;

                if (i < j) {
                    t1 = x[i];
                    x[i] = x[j];
                    x[j] = t1;
                    t1 = y[i];
                    y[i] = y[j];
                    y[j] = t1;
                }
            }

            n1 = 0;
            n2 = 1;
            for (i = 0; i < m; i++) {
                n1 = n2;
                n2 = n2 + n2;
                a = 0;

                for (j = 0; j < n1; j++) {
                    c = cos[a];
                    s = sin[a];
                    a += 1 << (m - i - 1);

                    for (k = j; k < n; k = k + n2) {
                        t1 = c * x[k + n1] - s * y[k + n1];
                        t2 = s * x[k + n1] + c * y[k + n1];
                        x[k + n1] = x[k] - t1;
                        y[k + n1] = y[k] - t2;
                        x[k] = x[k] + t1;
                        y[k] = y[k] + t2;
                    }
                }
            }
        }

    }

    class TrainingTuple {
        public ArrayList<Double> vals;
        public String label;
        public TrainingTuple(ArrayList<Double> vals, String label) {
            this.vals = vals;
            this.label = label;
        }
    }

    private class OmaInstance extends DenseInstance {

        public OmaInstance(double weight, double[] attValues) {
            super(weight, attValues);
        }

        @Override
        public/* @pure@ */boolean isMissing(int attIndex) {
            if (m_AttValues.length < attIndex) {
                return true;
            }
            return false;
        }
    }
}