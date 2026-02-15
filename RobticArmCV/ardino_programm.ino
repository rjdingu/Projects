#include <Servo.h>

Servo baseServo;
Servo shoulderServo;
Servo elbowServo;
Servo gripperServo;

int baseA = 90;
int shoulderA = 90;
int elbowA = 90;
int gripperA = 40;

void setup() {
  Serial.begin(9600);

  baseServo.attach(9);
  shoulderServo.attach(10);
  elbowServo.attach(11);
  gripperServo.attach(12);

  baseServo.write(baseA);
  shoulderServo.write(shoulderA);
  elbowServo.write(elbowA);
  gripperServo.write(gripperA);

  delay(1000);
}

void loop() {

  if (Serial.available()) {

    int b = Serial.parseInt();
    int s = Serial.parseInt();
    int e = Serial.parseInt();
    int g = Serial.parseInt();

    baseA = constrain(b, 15, 165);
    shoulderA = constrain(s, 25, 155);
    elbowA = constrain(e, 25, 155);
    gripperA = constrain(g, 30, 80);

    baseServo.write(baseA);
    shoulderServo.write(shoulderA);
    elbowServo.write(elbowA);
    gripperServo.write(gripperA);
  }
}
