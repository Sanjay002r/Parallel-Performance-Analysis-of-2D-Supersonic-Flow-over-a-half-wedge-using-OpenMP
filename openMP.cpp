#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>
#include <omp.h>
using namespace std;

// Constants
const int Imax = 97;
const int Jmax = 65;
const double Lx = 1.0, Ly = 1.0;
const double Wsp = Lx * 0.2;            // Wedge start point in domain
const double theta = 15 * M_PI / 180.0; // Wedge angle in radians
const double R = 287.04;                // Gas constant
const double gamma_val = 1.4;           // Specific heat ratio
const double CFL = 0.8;                 // CFL number
const double dist = Lx - Wsp;               

// Grid spacing
const double d_zeta = 1.0 / Imax;
const double d_eta = 1.0 / Jmax;

// Vector type alias for arrays
using Matrix3D = std::vector<std::vector<std::vector<double>>>;
using Matrix = std::vector<std::vector<double>>;
using Vectors = std::vector<double>;

Vectors linspace(double start, double end, int num) {
    Vectors result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// Create zeta and eta arrays with linspace
Vectors zeta = linspace(0, 1, Imax);  // zeta from 0 to 1 with Imax points
Vectors eta = linspace(0, 1, Jmax);   // eta from 0 to 1 with Jmax points
// Create physical mesh by transforming to the physical domain
Matrix X(Imax, Vectors(Jmax));
Matrix Y(Imax, Vectors(Jmax));

// Function for initializing the computational mesh
void initializeMesh() {
    // Transfinite interpolation for physical mesh
    for (int i = 0; i < Imax; ++i) {
        for (int j = 0; j < Jmax; ++j) {
            X[i][j] = Lx * zeta[i];
            if (zeta[i] >= 0.2) {
                Y[i][j] = (1 - zeta[i]) * Ly * eta[j] + zeta[i] * ((Ly - dist * tan(theta)) * eta[j] + dist * tan(theta)) + (1-eta[j])*(Lx*zeta[i]-Wsp)*tan(theta)+eta[j]*Ly-((1-zeta[i])*eta[j]*Ly+zeta[i]*(1-eta[j])*dist*tan(theta)+zeta[i]*eta[j]*Ly);
            } else {
                Y[i][j] = Ly * eta[j];
            }
        }
    }
}

Matrix interpolateToNodes(const Matrix& data) {
    int Imax = data.size();
    int Jmax = data[0].size();
    Matrix data_interp(Imax + 1, std::vector<double>(Jmax + 1, 0.0));

    // Internal nodes (average of surrounding cells)
    for (int i = 1; i < Imax; ++i) {
        for (int j = 1; j < Jmax; ++j) {
            data_interp[i][j] = 0.25 * (data[i-1][j-1] + data[i][j-1] + data[i-1][j] + data[i][j]);
        }
    }

    // Edges (average of adjacent cells)
    for (int j = 1; j < Jmax; ++j) {
        data_interp[0][j] = 0.5 * (data[0][j-1] + data[0][j]);          // Bottom edge
        data_interp[Imax][j] = 0.5 * (data[Imax-1][j-1] + data[Imax-1][j]); // Top edge
    }
    for (int i = 1; i < Imax; ++i) {
        data_interp[i][0] = 0.5 * (data[i-1][0] + data[i][0]);         // Left edge
        data_interp[i][Jmax] = 0.5 * (data[i-1][Jmax-1] + data[i][Jmax-1]); // Right edge
    }

    // Corners (copy nearest cell data)
    data_interp[0][0] = data[0][0];
    data_interp[Imax][0] = data[Imax-1][0];
    data_interp[0][Jmax] = data[0][Jmax-1];
    data_interp[Imax][Jmax] = data[Imax-1][Jmax-1];

    return data_interp;
}

void writeToVTKFile(const Matrix& X, const Matrix& Y, const Matrix& pressure, const Matrix& mach, const std::string& filename) {
    int Imax = X.size();
    int Jmax = X[0].size();

    // Interpolate data to nodes
    Matrix pressure_nodes = interpolateToNodes(pressure);
    Matrix mach_nodes = interpolateToNodes(mach);

    // Create a structured grid
    vtkSmartPointer<vtkStructuredGrid> structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
    structuredGrid->SetDimensions(Imax, Jmax, 1);

    // Create points and populate with X, Y, Z coordinates (Z = 0 for 2D)
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints((Imax) * (Jmax));
    for (int j = 0; j < Jmax; ++j) {
        for (int i = 0; i < Imax; ++i) {
            points->SetPoint(j * (Imax) + i, X[i][j], Y[i][j], 0.0); // Z = 0.0 for 2D
        }
    }
    structuredGrid->SetPoints(points);

    // Add pressure data
    vtkSmartPointer<vtkDoubleArray> pressureArray = vtkSmartPointer<vtkDoubleArray>::New();
    pressureArray->SetName("Pressure");
    pressureArray->SetNumberOfComponents(1);
    pressureArray->SetNumberOfTuples((Imax) * (Jmax));
    for (int j = 0; j < Jmax; ++j) {
        for (int i = 0; i < Imax; ++i) {
            pressureArray->SetValue(j * (Imax) + i, pressure_nodes[i][j]);
        }
    }
    structuredGrid->GetPointData()->AddArray(pressureArray);

    // Add Mach number data
    vtkSmartPointer<vtkDoubleArray> machArray = vtkSmartPointer<vtkDoubleArray>::New();
    machArray->SetName("Mach");
    machArray->SetNumberOfComponents(1);
    machArray->SetNumberOfTuples((Imax) * (Jmax));
    for (int j = 0; j < Jmax; ++j) {
        for (int i = 0; i < Imax; ++i) {
            machArray->SetValue(j * (Imax) + i, mach_nodes[i][j]);
        }
    }
    structuredGrid->GetPointData()->AddArray(machArray);

    // Write the structured grid to a VTK file
    vtkSmartPointer<vtkXMLStructuredGridWriter> writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
    std::string outputFilename = filename + ".vts";
    writer->SetFileName(outputFilename.c_str());
    writer->SetInputData(structuredGrid);
    writer->Write();
}

// Helper function for initializing 2D arrays
Matrix createMatrix(int rows, int cols, double value = 0.0){
    return Matrix(rows, Vectors(cols, value));
}

// Initialize matrices for the control volume parameters
Matrix Omega = createMatrix(Imax - 1, Jmax - 1);
Matrix dx1 = createMatrix(Imax - 1, Jmax - 1), dx2 = createMatrix(Imax - 1, Jmax - 1), dx3 = createMatrix(Imax - 1, Jmax - 1), dx4 = createMatrix(Imax - 1, Jmax - 1);
Matrix dy1 = createMatrix(Imax - 1, Jmax - 1), dy2 = createMatrix(Imax - 1, Jmax - 1), dy3 = createMatrix(Imax - 1, Jmax - 1), dy4 = createMatrix(Imax - 1, Jmax - 1);
Matrix del_S1 = createMatrix(Imax - 1, Jmax - 1), del_S2 = createMatrix(Imax - 1, Jmax - 1), del_S3 = createMatrix(Imax - 1, Jmax - 1), del_S4 = createMatrix(Imax - 1, Jmax - 1);
Matrix n1_x = createMatrix(Imax - 1, Jmax - 1), n1_y = createMatrix(Imax - 1, Jmax - 1), n2_x = createMatrix(Imax - 1, Jmax - 1), n2_y = createMatrix(Imax - 1, Jmax - 1);
Matrix n3_x = createMatrix(Imax - 1, Jmax - 1), n3_y = createMatrix(Imax - 1, Jmax - 1), n4_x = createMatrix(Imax - 1, Jmax - 1), n4_y = createMatrix(Imax - 1, Jmax - 1);
Matrix A1 = createMatrix(Imax - 1, Jmax - 1), A2 = createMatrix(Imax - 1, Jmax - 1), T1_x = createMatrix(Imax - 1, Jmax - 1), T1_y = createMatrix(Imax - 1, Jmax - 1);
Matrix T2_x = createMatrix(Imax - 1, Jmax - 1), T2_y = createMatrix(Imax - 1, Jmax - 1), Xc = createMatrix(Imax - 1, Jmax - 1), Yc = createMatrix(Imax - 1, Jmax - 1);

// Function to compute control volume parameters
void computeControlVolumeParameters(const Matrix& X, const Matrix& Y) {
    for (int i = 0; i < Imax - 1; i++) {
        for (int j = 0; j < Jmax - 1; j++) {
            Omega[i][j] = ((X[i + 1][j + 1] - X[i][j]) * (Y[i][j + 1] - Y[i + 1][j]) - (X[i][j + 1] - X[i + 1][j]) * (Y[i + 1][j + 1] - Y[i][j])) / 2.0;

            dx1[i][j] = X[i + 1][j] - X[i][j];
            dy1[i][j] = Y[i + 1][j] - Y[i][j];
            dx2[i][j] = X[i + 1][j + 1] - X[i + 1][j];
            dy2[i][j] = Y[i + 1][j + 1] - Y[i + 1][j];
            dx3[i][j] = X[i][j + 1] - X[i + 1][j + 1];
            dy3[i][j] = Y[i][j + 1] - Y[i + 1][j + 1];
            dx4[i][j] = X[i][j] - X[i][j + 1];
            dy4[i][j] = Y[i][j] - Y[i][j + 1];

            del_S1[i][j] = std::sqrt(dx1[i][j] * dx1[i][j] + dy1[i][j] * dy1[i][j]);
            del_S2[i][j] = std::sqrt(dx2[i][j] * dx2[i][j] + dy2[i][j] * dy2[i][j]);
            del_S3[i][j] = std::sqrt(dx3[i][j] * dx3[i][j] + dy3[i][j] * dy3[i][j]);
            del_S4[i][j] = std::sqrt(dx4[i][j] * dx4[i][j] + dy4[i][j] * dy4[i][j]);

            n1_x[i][j] = dy1[i][j] / del_S1[i][j];
            n1_y[i][j] = -dx1[i][j] / del_S1[i][j];
            n2_x[i][j] = dy2[i][j] / del_S2[i][j];
            n2_y[i][j] = -dx2[i][j] / del_S2[i][j];
            n3_x[i][j] = dy3[i][j] / del_S3[i][j];
            n3_y[i][j] = -dx3[i][j] / del_S3[i][j];
            n4_x[i][j] = dy4[i][j] / del_S4[i][j];
            n4_y[i][j] = -dx4[i][j] / del_S4[i][j];

            A1[i][j] = (X[i][j] * (Y[i][j + 1] - Y[i + 1][j]) + X[i][j + 1] * (Y[i + 1][j] - Y[i][j]) + X[i + 1][j] * (Y[i][j] - Y[i][j + 1])) / 2.0;
            A2[i][j] = (X[i][j + 1] * (Y[i + 1][j + 1] - Y[i + 1][j]) + X[i + 1][j + 1] * (Y[i + 1][j] - Y[i][j + 1]) + X[i + 1][j] * (Y[i][j + 1] - Y[i + 1][j + 1])) / 2.0;
            T1_x[i][j] = (X[i][j] + X[i][j + 1] + X[i + 1][j]) / 3.0;
            T1_y[i][j] = (Y[i][j] + Y[i][j + 1] + Y[i + 1][j]) / 3.0;
            T2_x[i][j] = (X[i][j + 1] + X[i + 1][j + 1] + X[i + 1][j]) / 3.0;
            T2_y[i][j] = (Y[i][j + 1] + Y[i + 1][j + 1] + Y[i + 1][j]) / 3.0;

            Xc[i][j] = (T1_x[i][j] * A1[i][j] + T2_x[i][j] * A2[i][j]) / (A1[i][j] + A2[i][j]);
            Yc[i][j] = (T1_y[i][j] * A1[i][j] + T2_y[i][j] * A2[i][j]) / (A1[i][j] + A2[i][j]);
        }
    }
}

// Function for conservative variable initialization
void conservative(Matrix3D &U){
    double p_init = 101353, T_init = 288.9, u_init = 852.4, v_init = 0.0, rho_init = p_init / (R * T_init);
    double e_init = p_init / (gamma_val - 1) + 0.5 * rho_init * (u_init * u_init + v_init * v_init);
    for (int i = 0; i < Imax - 1; i++){
        for (int j = 0; j < Jmax - 1; j++){
            U[0][i][j] = rho_init;
            U[1][i][j] = rho_init * u_init;
            U[2][i][j] = rho_init * v_init;
            U[3][i][j] = e_init;
        }
    }
}

void flux_primitives(Matrix3D &U_given, Matrix &rho, Matrix &p, Matrix &H, Matrix &u, Matrix &v){
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < U_given[0].size(); i++){
        for (int j = 0; j < U_given[0][0].size(); j++){
            rho[i][j] = U_given[0][i][j];
            u[i][j] = U_given[1][i][j] / rho[i][j];
            v[i][j] = U_given[2][i][j] / rho[i][j];
            p[i][j] = (gamma_val - 1) * (U_given[3][i][j] - 0.5 * rho[i][j] * (u[i][j] * u[i][j] + v[i][j] * v[i][j]));
            H[i][j] = (U_given[3][i][j] + p[i][j]) / rho[i][j];
        }
    }
}

Matrix3D compute_flux(const Matrix3D& U, const Matrix& nx, const Matrix& ny, int face) {
    Matrix3D Urip = U;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Urip[0].size(); i++) {
        for (int j = 0; j < Urip[0][0].size(); j++) {
            Urip[1][i][j] = U[1][i][j] * nx[i][j] + U[2][i][j] * ny[i][j];
            Urip[2][i][j] = -U[1][i][j] * ny[i][j] + U[2][i][j] * nx[i][j];
        }
    }

    Matrix3D Urip_l, Urip_r;
    if (face == 2) {
        Urip_l.resize(Urip.size(), Matrix(Urip[0].size() - 1, Vectors(Urip[0][0].size())));
        Urip_r.resize(Urip.size(), Matrix(Urip[0].size() - 1, Vectors(Urip[0][0].size())));
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < Urip[0].size() - 1; i++) {
                for (int j = 0; j < Urip[0][0].size(); j++) {
                    Urip_l[k][i][j] = Urip[k][i][j];
                    Urip_r[k][i][j] = Urip[k][i+1][j];
                }
            }
        }
    } else if (face == 4) {
        Urip_l.resize(Urip.size(), Matrix(Urip[0].size() - 1, Vectors(Urip[0][0].size())));
        Urip_r.resize(Urip.size(), Matrix(Urip[0].size() - 1, Vectors(Urip[0][0].size())));
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < Urip[0].size() - 1; i++) {
                for (int j = 0; j < Urip[0][0].size(); j++) {
                    Urip_l[k][i][j] = Urip[k][i+1][j];
                    Urip_r[k][i][j] = Urip[k][i][j];
                }
            }
        }
    } else if (face == 3) {
        Urip_l.resize(Urip.size(), Matrix(Urip[0].size(), Vectors(Urip[0][0].size() - 1)));
        Urip_r.resize(Urip.size(), Matrix(Urip[0].size(), Vectors(Urip[0][0].size() - 1)));
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < Urip[0].size(); i++) {
                for (int j = 0; j < Urip[0][0].size() - 1; j++) {
                    Urip_l[k][i][j] = Urip[k][i][j];
                    Urip_r[k][i][j] = Urip[k][i][j + 1];
                }
            }
        }
    } else {
        Urip_l.resize(Urip.size(), Matrix(Urip[0].size(), Vectors(Urip[0][0].size() - 1)));
        Urip_r.resize(Urip.size(), Matrix(Urip[0].size(), Vectors(Urip[0][0].size() - 1)));
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < Urip[0].size(); i++) {
                for (int j = 0; j < Urip[0][0].size()-1; j++) {
                    Urip_l[k][i][j] = Urip[k][i][j + 1];
                    Urip_r[k][i][j] = Urip[k][i][j];
                }
            }
        }
    }
    
    Matrix rho_l(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), p_l(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), H_l(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), u_n_l(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix u_t_l(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), rho_r(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), p_r(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), H_r(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix u_n_r(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), u_t_r(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix rho(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0)), p(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0)), H(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0)), u_n(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0)), u_t(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0));
    
    flux_primitives(Urip_l, rho_l, p_l, H_l, u_n_l, u_t_l);
    flux_primitives(Urip_r, rho_r, p_r, H_r, u_n_r, u_t_r);
    flux_primitives(Urip, rho, p, H, u_n, u_t);

    Matrix tilda_rho(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), temp(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix tilda_un(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), tilda_ut(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix tilda_H(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), tilda_a(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix delta_p(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), delta_rho(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), delta_un(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), delta_ut(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix alpha_1(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), alpha_2(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), alpha_3(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), alpha_4(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), lambda_1(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), lambda_2(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), lambda_3(Urip_l[0].size(), Vectors(Urip_l[0][0].size())), lambda_4(Urip_l[0].size(), Vectors(Urip_l[0][0].size()));
    Matrix3D K1(4, Matrix(Urip_l[0].size(), Vectors(Urip_l[0][0].size(), 1.0))), K2(4, Matrix(Urip_l[0].size(), Vectors(Urip_l[0][0].size(), 1.0))), K3(4, Matrix(Urip_l[0].size(), Vectors(Urip_l[0][0].size(), 0.0))), K4(4, Matrix(Urip_l[0].size(), Vectors(Urip_l[0][0].size(), 1.0)));
 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Urip_l[0].size(); i++) {
        for (int j = 0; j < Urip_l[0][0].size(); j++) {
            tilda_rho[i][j] = std::sqrt(rho_l[i][j] * rho_r[i][j]);
            temp[i][j] = std::sqrt(rho_l[i][j]) / (std::sqrt(rho_l[i][j]) + std::sqrt(rho_r[i][j]));
            tilda_un[i][j] = temp[i][j] * u_n_l[i][j] + (1 - temp[i][j]) * u_n_r[i][j];
            tilda_ut[i][j] = temp[i][j] * u_t_l[i][j] + (1 - temp[i][j]) * u_t_r[i][j];
            tilda_H[i][j] = temp[i][j] * H_l[i][j] + (1 - temp[i][j]) * H_r[i][j];
            tilda_a[i][j] = std::sqrt((gamma_val - 1) * (tilda_H[i][j] - (tilda_un[i][j] * tilda_un[i][j] + tilda_ut[i][j] * tilda_ut[i][j]) / 2));
            delta_p[i][j] = p_r[i][j] - p_l[i][j];
            delta_rho[i][j] = rho_r[i][j] - rho_l[i][j];
            delta_un[i][j] = u_n_r[i][j] - u_n_l[i][j];
            delta_ut[i][j] = u_t_r[i][j] - u_t_l[i][j];

            alpha_1[i][j] = (delta_p[i][j] + tilda_rho[i][j] * tilda_a[i][j] * delta_un[i][j]) / (2 * tilda_a[i][j] * tilda_a[i][j]);
            alpha_2[i][j] = delta_rho[i][j] - (delta_p[i][j] / (tilda_a[i][j] * tilda_a[i][j]));
            alpha_3[i][j] = tilda_rho[i][j] * delta_ut[i][j];
            alpha_4[i][j] = (delta_p[i][j] - tilda_rho[i][j] * tilda_a[i][j] * delta_un[i][j]) / (2 * tilda_a[i][j] * tilda_a[i][j]);

            lambda_1[i][j] = std::abs(tilda_un[i][j] + tilda_a[i][j]);
            lambda_2[i][j] = std::abs(tilda_un[i][j]);
            lambda_3[i][j] = std::abs(tilda_un[i][j]);
            lambda_4[i][j] = std::abs(tilda_un[i][j] - tilda_a[i][j]);

            K1[1][i][j] = tilda_un[i][j] + tilda_a[i][j];
            K1[2][i][j] = tilda_ut[i][j];
            K1[3][i][j] = tilda_H[i][j] + tilda_un[i][j] * tilda_a[i][j];

            K2[1][i][j] = tilda_un[i][j];
            K2[2][i][j] = tilda_ut[i][j];
            K2[3][i][j] = (tilda_un[i][j] * tilda_un[i][j] + tilda_ut[i][j] * tilda_ut[i][j]) / 2;

            K3[2][i][j] = 1.0;
            K3[3][i][j] = tilda_ut[i][j];

            K4[1][i][j] = tilda_un[i][j] - tilda_a[i][j];
            K4[2][i][j] = tilda_ut[i][j];
            K4[3][i][j] = tilda_H[i][j] - tilda_un[i][j] * tilda_a[i][j];
        }
    }    
    
    Matrix3D F_1D(4, Matrix(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0))), F(4, Matrix(Urip[0].size(), Vectors(Urip[0][0].size(), 1.0))),F_out(4, Matrix(Imax - 1, Vectors(Jmax - 1)));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Urip[0].size(); i++) {
        for (int j = 0; j < Urip[0][0].size(); j++) {
          F_1D[0][i][j] = rho[i][j] * u_n[i][j];
          F_1D[1][i][j] = rho[i][j] * u_n[i][j] * u_n[i][j] + p[i][j];
          F_1D[2][i][j] = rho[i][j] * u_n[i][j] * u_t[i][j];
          F_1D[3][i][j] = (Urip[3][i][j] + p[i][j]) * u_n[i][j];
        }
    }

    if (face == 1) {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < 4; i++) {
            for (int k =0; k < Imax -1; k++){
                for (int j = 0; j < Jmax - 2; j++) {
                   F[i][k][j + 1] = 0.5 * (F_1D[i][k][j] + F_1D[i][k][j + 1]) - 0.5 * (alpha_1[k][j] * K1[i][k][j] * lambda_1[k][j] + alpha_2[k][j] * K2[i][k][j] * lambda_2[k][j] + alpha_3[k][j] * K3[i][k][j] * lambda_3[k][j] + alpha_4[k][j] * K4[i][k][j] * lambda_4[k][j]);
                }
            }
        }
    } else if (face == 2) {
        #pragma omp parallel for collapse(3)
        for (int i_f = 0; i_f < 4; i_f++){
            for (int i = 0; i < Imax - 2; i++) {
                for (int j = 0; j < Jmax - 1; j++) {
                    F[i_f][i][j] = 0.5 * (F_1D[i_f][i][j] + F_1D[i_f][i + 1][j]) - 0.5 * (alpha_1[i][j] * K1[i_f][i][j] * lambda_1[i][j] + alpha_2[i][j] * K2[i_f][i][j] * lambda_2[i][j] + alpha_3[i][j] * K3[i_f][i][j] * lambda_3[i][j] + alpha_4[i][j] * K4[i_f][i][j] * lambda_4[i][j]);
                }
            }
        }
    } else if (face == 3) {
        #pragma omp parallel for collapse(3)
        for (int i_f = 0; i_f < 4; i_f++){
            for (int i = 0; i < Imax - 1; i++){
                for (int j = 0; j < Jmax - 2; j++) {
                    F[i_f][i][j] = 0.5 * (F_1D[i_f][i][j] + F_1D[i_f][i][j + 1]) - 0.5 * (alpha_1[i][j] * K1[i_f][i][j] * lambda_1[i][j] + alpha_2[i][j] * K2[i_f][i][j] * lambda_2[i][j] + alpha_3[i][j] * K3[i_f][i][j] * lambda_3[i][j] + alpha_4[i][j] * K4[i_f][i][j] * lambda_4[i][j]);
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(3)
        for (int i_f = 0; i_f < 4; i_f++){
            for (int i = 0; i < Imax - 2; i++){
                for (int j = 0; j < Jmax - 1; j++){
                    F[i_f][i + 1][j] = 0.5 * (F_1D[i_f][i][j] + F_1D[i_f][i + 1][j]) - 0.5 * (alpha_1[i][j] * K1[i_f][i][j] * lambda_1[i][j] + alpha_2[i][j] * K2[i_f][i][j] * lambda_2[i][j] + alpha_3[i][j] * K3[i_f][i][j] * lambda_3[i][j] + alpha_4[i][j] * K4[i_f][i][j] * lambda_4[i][j]);
                }
            }
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Imax - 1; i++) {
        for (int j = 0; j < Jmax - 1; j++) {
            F_out[0][i][j] = F[0][i][j];
            F_out[1][i][j] = F[1][i][j] * nx[i][j] - F[2][i][j] * ny[i][j];
            F_out[2][i][j] = F[1][i][j] * ny[i][j] + F[2][i][j] * nx[i][j];
            F_out[3][i][j] = F[3][i][j];
        }
    }
    return F_out;
}

void Roe_Flux_Scheme(const Matrix3D &U, Matrix3D &F1, Matrix3D &F2, Matrix3D &F3, Matrix3D &F4, double &dt) {
    Matrix rho= createMatrix(Imax - 1, Jmax - 1),u= createMatrix(Imax - 1, Jmax - 1),v= createMatrix(Imax - 1, Jmax - 1),e= createMatrix(Imax - 1, Jmax - 1),p= createMatrix(Imax - 1, Jmax - 1),a= createMatrix(Imax - 1, Jmax - 1),lambda_x= createMatrix(Imax - 1, Jmax - 1),lambda_y= createMatrix(Imax - 1, Jmax - 1);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Imax - 1; i++) {
        for (int j = 0; j < Jmax - 1; j++) {
            rho[i][j] = U[0][i][j];
            u[i][j] = U[1][i][j] / rho[i][j];
            v[i][j] = U[2][i][j] / rho[i][j];
            e[i][j] = U[3][i][j];
            p[i][j] = (gamma_val - 1) * (e[i][j] - 0.5 * rho[i][j] * (u[i][j] * u[i][j] + v[i][j] * v[i][j]));
            a[i][j] = std::sqrt(gamma_val * p[i][j] / rho[i][j]);

            lambda_x[i][j] = (std::fabs(u[i][j] * (n2_x[i][j] - n4_x[i][j]) / 2 + v[i][j] * (n2_y[i][j] - n4_y[i][j]) / 2) + a[i][j]) * (del_S4[i][j] + del_S2[i][j]) / 2;
            lambda_y[i][j] = (std::fabs(u[i][j] * (n3_x[i][j] - n1_x[i][j]) / 2 + v[i][j] * (n3_y[i][j] - n1_y[i][j]) / 2) + a[i][j]) * (del_S3[i][j] + del_S1[i][j]) / 2;
            dt = std::min(dt, CFL * Omega[i][j] / (lambda_x[i][j] + lambda_y[i][j]));
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        F1 = compute_flux(U, n1_x, n1_y, 1);
        #pragma omp section
        F2 = compute_flux(U, n2_x, n2_y, 2);
        #pragma omp section
        F3 = compute_flux(U, n3_x, n3_y, 3);
        #pragma omp section
        F4 = compute_flux(U, n4_x, n4_y, 4);
    }

    #pragma omp parallel for collapse(1)
    for (int j = 0; j < F1[0].size(); j++) {
        F1[0][j][0] = 0; 
        F1[1][j][0] = p[j][0] * n1_x[j][0];  // boundary for normal flux component
        F1[2][j][0] = p[j][0] * n1_y[j][0];  // boundary for tangential flux component
        F1[3][j][0] = 0;
    }
}

double calculate_RSS(const Matrix3D& Unew, const Matrix3D& U) {
    double RSS = 0.0;
    for (int i = 0; i < Unew[0].size(); ++i) {
        for (int j = 0; j < Unew[0][0].size(); ++j) {
            RSS += std::pow((Unew[0][i][j] - U[0][i][j]) / U[0][i][j], 2);
        }
    }
    return std::sqrt(RSS / (Imax * Jmax));
}

void solve(Matrix3D &U) {
    double t = 0;
    Matrix3D Unew(4, Matrix(Imax - 1, Vectors((Jmax - 1), 0.0)));
    double RSS = 1.0;
    Matrix history = {{t, RSS}};

    while (RSS >= 1e-6) {
        Matrix3D F1, F2, F3, F4;
        double dt = 100;
        Roe_Flux_Scheme(U, F1, F2, F3, F4, dt);

        // Updating Unew for all interior points
        #pragma omp parallel for collapse(3)
        for (int I = 1; I < Imax - 2; I++) {
            for (int J = 0; J < Jmax -1 ; J++) {
                for (int k = 0; k < U.size(); k++) {
                    Unew[k][I][J] = U[k][I][J] - dt / Omega[I][J] * (F1[k][I][J] * del_S1[I][J] + F2[k][I][J] * del_S2[I][J] + F3[k][I][J] * del_S3[I][J] + F4[k][I][J] * del_S4[I][J]);
                }
            }
        }

        // Fixed Inlet
        for (int k = 0; k < U.size(); k++) {
            Unew[k][0] = U[k][0];
        }

        // Right boundary zero-gradient Outlet
        for (int k = 0; k < U.size(); k++) {
            for (int J = 0; J < Jmax - 1; J++) {
                Unew[k][Imax - 2][J] = Unew[k][Imax - 3][J];
            }
        }

       // Fixed Freestream top
        for (int I = 0; I < Imax - 1; I++) {
            for (int k = 0; k < U.size(); k++) {
                Unew[k][I][Jmax -2] = U[k][I][Jmax -2];
                
            }
        }

        std::cout << "dt " << dt << std::endl;
        t += dt;
        RSS = calculate_RSS(Unew, U);
        std::cout << history.back()[1] << std::endl;
        history.push_back({t, RSS});
        U = Unew;

    }
}

int main(){
    Matrix3D U(4, Matrix(Imax - 1, Vectors(Jmax - 1)));

    initializeMesh();
    computeControlVolumeParameters(X, Y);
    conservative(U);
    solve(U);

    Matrix rho(Imax -1, Vectors(Jmax -1, 0.0)), p(Imax -1, Vectors(Jmax -1, 0.0)), H(Imax -1, Vectors(Jmax -1, 0.0)), u(Imax -1, Vectors(Jmax -1, 0.0)), v(Imax -1, Vectors(Jmax -1, 0.0)),M(Imax -1, Vectors(Jmax -1, 0.0));
    flux_primitives(U, rho, p, H, u, v);

    for (int i = 0; i < Imax - 1; ++i) {
        for (int j = 0; j < Jmax - 1; ++j) {
            double velocity = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
            double sound_speed = std::sqrt(gamma_val * p[i][j] / rho[i][j]);
            M[i][j] = velocity / sound_speed;
        }
    }
    writeToVTKFile(X, Y, p, M, "OutputGrid");

}