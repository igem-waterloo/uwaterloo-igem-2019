// Gmsh project created on Tue Aug 27 00:16:49 2019
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-4, 0, 0, 8, -10, 0};
//+
Circle(5) = {0, -3, 0, 0.5, 0, 2*Pi};
//+
Circle(6) = {0, -5, 0, 0.5, 0, 2*Pi};
//+
Curve{5} In Surface{1};
//+
Curve{5} In Surface{1};
//+
Curve{6} In Surface{1};
