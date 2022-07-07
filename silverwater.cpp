Mat Lane_warning(Mat img_input, vector<Point> lane) {
   Vec2i R1 = lane[0];
   Vec2i R2 = lane[1];
   Vec2i L1 = lane[2];
   Vec2i L2 = lane[3];

   printf("R1: (%d, %d)", R1[0], R1[1]);
   printf(" R2: (%d, %d)", R2[0], R2[1]);
   printf(" L1: (%d, %d)", L1[0], L1[1]);
   printf(" L2: (%d, %d)\n", L2[0], L2[1]);

   double lane_center = (R1[0] + L1[0]) / 2;
   printf("lane_center %f\n", lane_center);
   printf("lane_center %f\n", img_center);

   putText(img_input, "|", Point(img_center, 450), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
   putText(img_input, "|", Point(lane_center, 450), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);

   if (lane_center < (img_center - 50)) {
      printf("Warning\n");
      putText(img_input, "CAUTION", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
   }
   else if (lane_center > (img_center + 50)) {
      printf("Warning\n");
      putText(img_input, "CAUTION", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
   }
   else {
      printf("Safe\n");
      putText(img_input, "SAFE", Point(20, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);
   }

   return img_input;
}
