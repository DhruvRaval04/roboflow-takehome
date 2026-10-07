/**********************************************************************
  Filename    : CameraStation (from Freenove Sketch_07.1 Camera Web Server)
  Description : The camera images captured by the ESP32S3 are displayed on the web page.
  Auther      : www.freenove.com
  Modification: 2026/05/16
**********************************************************************/
#include "esp_camera.h"
#include <WiFi.h>
#include <esp_wifi.h>
#include "board_config.h"
// ===================
// Select camera model
// ===================
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE // Has PSRAM
#define CAMERA_MODEL_ESP32S3_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_UNITCAM // No PSRAM
//#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM
// ** Espressif Internal Boards **
//#define CAMERA_MODEL_ESP32_CAM_BOARD
//#define CAMERA_MODEL_ESP32S2_CAM_BOARD
//#define CAMERA_MODEL_ESP32S3_CAM_LCD

#include "camera_pins.h"

// ===========================================================================
// STATION mode: the camera joins the home router (dlink-BD84, 2.4 GHz) so the
// laptop can pull frames while keeping its own internet connection.
//
//   ESP32 --WiFi--> router <--WiFi-- laptop (Python: cv2.VideoCapture(stream))
//
// Credentials live in wifi_secrets.h (gitignored, never committed) -- copy
// wifi_secrets.example.h to wifi_secrets.h and fill in the password.
//
// FALLBACK: if the router join fails within JOIN_TIMEOUT_MS, the board starts
// the old 'Sunshine' access point instead (camera at http://192.168.4.1), so a
// wrong password never leaves the camera unreachable -- your phone still works.
// ===========================================================================
#include <ESPmDNS.h>       // advertises "rccam.local" so the laptop needn't know the DHCP IP
#include "wifi_secrets.h"  // defines WIFI_SSID and WIFI_PASSWORD

const char* AP_SSID = "Sunshine";
const char* AP_PASS = "Sunshine";
const unsigned long JOIN_TIMEOUT_MS = 20000;  // DHCP on a home router takes ~2-5 s

camera_config_t config;

void startCameraServer();
void camera_init();

// Try to join the router. Returns true once we hold a DHCP lease.
bool joinRouter() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Joining '%s'", WIFI_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < JOIN_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  return WiFi.status() == WL_CONNECTED;
}

void startFallbackAP() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_AP);
  // Same AP settings as Sketch_07.1 (WPA2/CCMP, no PMF, b/g only) -- they were
  // tuned so the Pico W's old CYW43 driver could associate.
  WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 4, false, WIFI_AUTH_WPA2_PSK, WIFI_CIPHER_TYPE_CCMP);
  esp_wifi_disable_pmf_config(WIFI_IF_AP);
  esp_wifi_set_protocol(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G);
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_init();

  bool onRouter = joinRouter();
  if (!onRouter) {
    Serial.println("Router join FAILED (wrong password? 5 GHz-only SSID?) -> fallback AP 'Sunshine'");
    startFallbackAP();
  }
  // Modem sleep makes the radio doze between beacons -> stream stutters and
  // laggy frames reach the detector. Streaming wants the radio always awake.
  WiFi.setSleep(false);

  if (onRouter && MDNS.begin("rccam")) {
    MDNS.addService("http", "tcp", 80);  // -> http://rccam.local works on Windows 10+
  }

  startCameraServer();  // UI on :80, raw MJPEG stream on :81/stream

  IPAddress ip = onRouter ? WiFi.localIP() : WiFi.softAPIP();
  // Python reads this exact line over serial to learn the address.
  Serial.printf("CAMERA_READY mode=%s ip=%s stream=http://%s:81/stream\n",
                onRouter ? "STA" : "AP", ip.toString().c_str(), ip.toString().c_str());
}

void loop() {
  // Camera work happens in the web-server tasks. Every 5 s, restate where we
  // are, so anyone opening the serial port late still learns the IP.
  if (WiFi.getMode() == WIFI_STA) {
    Serial.printf("STA ip=%s rssi=%d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    Serial.printf("AP ip=%s stations=%d\n", WiFi.softAPIP().toString().c_str(), WiFi.softAPgetStationNum());
  }
  delay(5000);
}

void camera_init() {
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 1;
  
  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    if(err==ESP_ERR_NOT_SUPPORTED){
      config.pixel_format = PIXFORMAT_RGB565;
      esp_err_t err = esp_camera_init(&config);
      if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
      }
    }
  }

  sensor_t * s = esp_camera_sensor_get();
  // drop down frame size for higher initial frame rate
  uint16_t pid = s->id.PID;
  if(pid == OV2640_PID){
    s->set_hmirror(s, 1);
    s->set_vflip(s, 1);     
  }
  else if(pid == OV3660_PID){
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);     
  }
  else if(pid == GC2145_PID){
    s->set_hmirror(s, 0);
    delay(500);
    s->set_vflip(s, 0);      
  }
  else if(pid == GC0308_PID){
    s->set_hmirror(s, 0);
    delay(500);
    s->set_vflip(s, 0);     
  }
  else{
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);       
  }
  s->set_brightness(s, 1);  // Slightly increase brightness
  s->set_saturation(s, 0);  // Reduce saturation
  s->set_ae_level(s, -3);   // Set exposure compensation level
}