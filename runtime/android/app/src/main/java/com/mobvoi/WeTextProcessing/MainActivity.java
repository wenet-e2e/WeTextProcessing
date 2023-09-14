package com.mobvoi.WeTextProcessing;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

  private static final String LOG_TAG = "WETEXTPROCESSING";
  private static final List<String> resource = Arrays.asList(
    "zh_tn_tagger.fst", "zh_tn_verbalizer.fst", "zh_itn_tagger.fst", "zh_itn_verbalizer.fst"
  );

  public static void assetsInit(Context context) throws IOException {
    AssetManager assetMgr = context.getAssets();
    // Unzip all files in resource from assets to context.
    // Note: Uninstall the APP will remove the resource files in the context.
    for (String file : assetMgr.list("")) {
      if (resource.contains(file)) {
        File dst = new File(context.getFilesDir(), file);
        if (!dst.exists() || dst.length() == 0) {
          Log.i(LOG_TAG, "Unzipping " + file + " to " + dst.getAbsolutePath());
          InputStream is = assetMgr.open(file);
          OutputStream os = new FileOutputStream(dst);
          byte[] buffer = new byte[4 * 1024];
          int read;
          while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
          }
          os.flush();
        }
      }
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    try {
      assetsInit(this);
    } catch (IOException e) {
      Log.e(LOG_TAG, "Error process asset files to file path");
    }

    TextView textView = findViewById(R.id.textView);
    textView.setText("");
    WeTextProcessing.init(getFilesDir().getPath());

    EditText editInput = findViewById(R.id.editTextInput);

    Button buttonNormalize = findViewById(R.id.button);
    buttonNormalize.setOnClickListener(view -> {
      String result = WeTextProcessing.normalize(editInput.getText().toString());
      textView.setText(result);
    });

    Button buttonInverseNormalize = findViewById(R.id.button2);
    buttonInverseNormalize.setOnClickListener(view -> {
      String result = WeTextProcessing.inverse_normalize(editInput.getText().toString());
      textView.setText(result);
    });
  }
}