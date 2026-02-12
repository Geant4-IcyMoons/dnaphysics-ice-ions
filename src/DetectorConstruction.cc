//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file DetectorConstruction.cc
/// \brief Implementation of the DetectorConstruction class

#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "G4VModularPhysicsList.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4NistManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace {
std::string ToLower(std::string s)
{
  for (auto& ch : s) ch = static_cast<char>(std::tolower(ch));
  return s;
}

bool IsIcePhysicsEnabled()
{
  const char* env = std::getenv("DNA_PHYSICS");
  if (!env || !*env) return true;  // project default is ice
  const std::string val = ToLower(std::string(env));
  return val != "water";
}

}  // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

DetectorConstruction::DetectorConstruction(G4VModularPhysicsList* ptr)
  : G4VUserDetectorConstruction(),
    fpWaterMaterial(nullptr),
    fpWorldMaterial(nullptr),
    fLogicWorld(nullptr),
    fLogicIce(nullptr),
    fPhysiWorld(nullptr),
    fPhysiIce(nullptr)
{
  // Create commands for interactive definition of the detector
  fDetectorMessenger = new DetectorMessenger(this, ptr);

  // Default values
  //
  // Ice size (cube by default)
  fIceSizeX = 100 * um;
  fIceSizeY = 100 * um;
  fIceSizeZ = 100 * um;
  fWorldSize = 0.;
  // and material
  G4NistManager* man = G4NistManager::Instance();
  if (IsIcePhysicsEnabled()) {
    fpWaterMaterial = man->FindOrBuildMaterial("G4_ICE");
    if (!fpWaterMaterial) fpWaterMaterial = man->FindOrBuildMaterial("G4_WATER");
  } else {
    fpWaterMaterial = man->FindOrBuildMaterial("G4_WATER");
  }
  fpWorldMaterial = man->FindOrBuildMaterial("G4_Galactic");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

DetectorConstruction::~DetectorConstruction()
{
  delete fDetectorMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void DetectorConstruction::DefineMaterials()
{
  // Water is defined from NIST material database
  G4NistManager* man = G4NistManager::Instance();

  // Some DNA models query G4_WATER internally even when the target is ice.
  // Ensure it is present in the material table to avoid null lookups.
  man->FindOrBuildMaterial("G4_WATER");
  G4Material* H2O = nullptr;
  if (IsIcePhysicsEnabled()) {
    H2O = man->FindOrBuildMaterial("G4_ICE");
  }
  if (!H2O) {
    H2O = man->FindOrBuildMaterial("G4_WATER");
  }
  G4Material* vacuum = man->FindOrBuildMaterial("G4_Galactic");

  /*
   If one wishes to test other density value for water material,
   one should use instead:

   G4Material * H2O = man->BuildMaterialWithNewDensity("G4_WATER_ICE",
   "G4_WATER",<density>*g/cm3);

   Note: any string for "G4_WATER_MODIFIED" parameter is accepted
   and "G4_WATER" parameter should not be changed
   Both materials are created and can be selected from dna.mac
   */

  if (!fpWaterMaterial) fpWaterMaterial = H2O;
  if (!fpWorldMaterial) {
    fpWorldMaterial = vacuum;
  }

  // G4cout << "-> Density of water material (g/cm3)="
  //  << fpWaterMaterial->GetDensity()/(g/cm/cm/cm) << G4endl;

  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Material*
DetectorConstruction::MaterialWithDensity(G4String name, G4double density)
{
  // Water is defined from NIST material database
  G4NistManager* man = G4NistManager::Instance();

  G4Material * material = man->BuildMaterialWithNewDensity(name,
   "G4_WATER", density);

   G4cout << "-> Density of water_modified material (g/cm3)="
    << material->GetDensity()/(g/cm/cm/cm) << G4endl;

 return material;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  DefineMaterials();

  G4cout << "DetectorConstruction: building Ice slab "
         << fIceSizeX / mm << " x " << fIceSizeY / mm << " x "
         << fIceSizeZ / mm << " mm"
         << ", material=" << (fpWaterMaterial ? fpWaterMaterial->GetName() : "NULL")
         << G4endl;

  // World volume: vacuum cube sized from ice dimensions.
  const G4double iceMax = std::max({fIceSizeX, fIceSizeY, fIceSizeZ});
  G4double worldSize = 1.5 * iceMax;
  // Ensure world contains an ice slab that starts at z=0.
  worldSize = std::max(worldSize, 2.0 * fIceSizeZ);
  fWorldSize = worldSize;

  G4Box* solidWorld = new G4Box("World",  // its name
                                worldSize / 2, worldSize / 2, worldSize / 2);
                                // its size

  fLogicWorld = new G4LogicalVolume(solidWorld,  // its solid
                                    fpWorldMaterial,  // its material
                                    "World");  // its name

  fPhysiWorld = new G4PVPlacement(0,  // no rotation
                                  G4ThreeVector(),  // at (0,0,0)
                                  "World",  // its name
                                  fLogicWorld,  // its logical volume
                                  0,  // its mother volume
                                  false,  // no boolean operation
                                  0);  // copy number

  // Visualization attributes - white
  G4VisAttributes* worldVisAtt = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0));
  worldVisAtt->SetVisibility(true);
  fLogicWorld->SetVisAttributes(worldVisAtt);

  // Ice slab placed from z=0 to z=+fIceSizeZ
  G4Box* solidIce = new G4Box("Ice", fIceSizeX / 2, fIceSizeY / 2, fIceSizeZ / 2);
  fLogicIce = new G4LogicalVolume(solidIce, fpWaterMaterial, "Ice");

  const G4ThreeVector iceCenter(0.0, 0.0, fIceSizeZ / 2.0);
  fPhysiIce = new G4PVPlacement(0, iceCenter, "Ice", fLogicIce, fPhysiWorld,
                                false, 0);

  G4VisAttributes* iceVis = new G4VisAttributes(G4Colour(0.0, 0.0, 1.0));
  iceVis->SetVisibility(true);
  fLogicIce->SetVisAttributes(iceVis);

  // Shows how to introduce a 20 eV tracking cut
  // logicWorld->SetUserLimits(new G4UserLimits(DBL_MAX,DBL_MAX,DBL_MAX,20*eV));

  return fPhysiWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetMaterial(const G4String& materialChoice)
{
  G4Material* pttoMaterial = nullptr;
  pttoMaterial = G4Material::GetMaterial(materialChoice, false);
  if (!pttoMaterial) {
    // Search the material by its name
    pttoMaterial = G4NistManager::Instance()->FindOrBuildMaterial(materialChoice);
  }

  if (pttoMaterial) {
    fpWaterMaterial = pttoMaterial;
    if (fLogicIce) {
      fLogicIce->SetMaterial(fpWaterMaterial);
    }
    G4RunManager::GetRunManager()->GeometryHasBeenModified();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetSize(G4double value)
{
  fIceSizeX = value;
  fIceSizeY = value;
  fIceSizeZ = value;
  G4cout << "DetectorConstruction: SetSize -> "
         << fIceSizeX / mm << " x " << fIceSizeY / mm << " x "
         << fIceSizeZ / mm << " mm" << G4endl;
  fPhysiWorld = nullptr;
  fPhysiIce = nullptr;
  fLogicWorld = nullptr;
  fLogicIce = nullptr;
  G4RunManager::GetRunManager()->ReinitializeGeometry(true);
}

void DetectorConstruction::SetIceSize(G4double x, G4double y, G4double z)
{
  fIceSizeX = x;
  fIceSizeY = y;
  fIceSizeZ = z;
  G4cout << "DetectorConstruction: SetIceSize -> "
         << fIceSizeX / mm << " x " << fIceSizeY / mm << " x "
         << fIceSizeZ / mm << " mm" << G4endl;
  fPhysiWorld = nullptr;
  fPhysiIce = nullptr;
  fLogicWorld = nullptr;
  fLogicIce = nullptr;
  G4RunManager::GetRunManager()->ReinitializeGeometry(true);
}
