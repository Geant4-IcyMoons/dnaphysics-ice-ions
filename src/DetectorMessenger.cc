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
/// \file DetectorMessenger.cc
/// \brief Implementation of the DetectorMessenger class

#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "RunAction.hh"

#include "G4GeneralParticleSourceData.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcommand.hh"
#include "G4UIdirectory.hh"
#include "G4UIparameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UImanager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <iomanip>
#include <sstream>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::DetectorMessenger(DetectorConstruction* Det, G4VModularPhysicsList* PL)
  : G4UImessenger(), fpDetector(Det), fpPhysList(PL)
{
  fpDetDir = new G4UIdirectory("/dna/test/");
  fpDetDir->SetGuidance("dna test commands");

  fpMaterCmd = new G4UIcmdWithAString("/dna/test/setMat", this);
  fpMaterCmd->SetGuidance("Select material of the world.");
  fpMaterCmd->SetParameterName("Material", false);
  fpMaterCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fpMaterCmd->SetToBeBroadcasted(false);

  fpPhysCmd = new G4UIcmdWithAString("/dna/test/addPhysics", this);
  fpPhysCmd->SetGuidance("Added Physics List");
  fpPhysCmd->SetParameterName("Physics", false);
  fpPhysCmd->AvailableForStates(G4State_PreInit);
  fpPhysCmd->SetToBeBroadcasted(false);

  fpTrackingCutCmd = new G4UIcmdWithABool("/dna/test/addIonsTrackingCut", this);
  fpTrackingCutCmd->SetGuidance("Added Ions Tracking Cut");
  fpTrackingCutCmd->SetDefaultValue(false);
  fpTrackingCutCmd->AvailableForStates(G4State_PreInit);
  fpTrackingCutCmd->SetToBeBroadcasted(false);

  fDensityCmd = new G4UIcommand("/dna/test/setMatDens",this);
  fDensityCmd->SetGuidance("Set density of the target material");
  G4UIparameter* symbPrm = new G4UIparameter("name",'s',false);
  symbPrm->SetGuidance("material name");
  fDensityCmd->SetParameter(symbPrm);
  G4UIparameter* densityPrm = new G4UIparameter("density",'d',false);
  densityPrm->SetGuidance("density of material");
  densityPrm->SetParameterRange("density>0.");
  fDensityCmd->SetParameter(densityPrm);
  G4UIparameter* unitPrm = new G4UIparameter("unit",'s',false);
  unitPrm->SetGuidance("unit of density");
  G4String unitList = G4UIcommand::UnitsList(G4UIcommand::CategoryOf("g/cm3"));
  unitPrm->SetParameterCandidates(unitList);
  fDensityCmd->SetParameter(unitPrm);
  fDensityCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fDensityCmd->SetToBeBroadcasted(false);

  fSizeCmd = new G4UIcmdWithADoubleAndUnit("/dna/test/setSize",this);
  fSizeCmd->SetGuidance("Set ice cube size (legacy, isotropic).");
  fSizeCmd->SetParameterName("Size",false);
  fSizeCmd->SetRange("Size>0.");
  fSizeCmd->SetUnitCategory("Length");
  fSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fSizeCmd->SetToBeBroadcasted(false);

  fIceSizeCmd = new G4UIcmdWith3VectorAndUnit("/dna/test/setIceSize", this);
  fIceSizeCmd->SetGuidance("Set ice slab size: X Y Z (full lengths).");
  fIceSizeCmd->SetParameterName("SizeX", "SizeY", "SizeZ", false);
  fIceSizeCmd->SetRange("SizeX>0. && SizeY>0. && SizeZ>0.");
  fIceSizeCmd->SetUnitCategory("Length");
  fIceSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fIceSizeCmd->SetToBeBroadcasted(false);

  fMaxThetaCmd = new G4UIcmdWithABool("/dna/test/setMaxTheta", this);
  fMaxThetaCmd->SetGuidance("Set cosine maxtheta mode.");
  fMaxThetaCmd->SetGuidance("true: compute geometry-limited /gps/ang/maxtheta.");
  fMaxThetaCmd->SetGuidance("false: set /gps/ang/maxtheta to 90 deg.");
  fMaxThetaCmd->SetDefaultValue(true);
  fMaxThetaCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fMaxThetaCmd->SetToBeBroadcasted(false);

  fLogModeCmd = new G4UIcmdWithAString("/dna/test/setLogMode", this);
  fLogModeCmd->SetGuidance("Set simulation logging mode: full or minimal.");
  fLogModeCmd->SetParameterName("mode", false);
  fLogModeCmd->SetCandidates("full minimal");
  fLogModeCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  fLogModeCmd->SetToBeBroadcasted(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::~DetectorMessenger()
{
  delete fpDetDir;

  delete fpMaterCmd;
  delete fpPhysCmd;
  delete fpTrackingCutCmd;
  delete fDensityCmd;
  delete fSizeCmd;
  delete fIceSizeCmd;
  delete fMaxThetaCmd;
  delete fLogModeCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fpMaterCmd) fpDetector->SetMaterial(newValue);

  if (command == fpPhysCmd) {
    auto* phys = dynamic_cast<PhysicsList*>(fpPhysList);
    if (phys != nullptr) {
      phys->AddPhysics(newValue);
    } else {
      G4cout << "### DetectorMessenger Warning: addPhysics is only supported for PhysicsList (ice)."
             << G4endl;
    }
  }

  if (command == fpTrackingCutCmd) {
    auto* phys = dynamic_cast<PhysicsList*>(fpPhysList);
    if (phys != nullptr) {
      phys->SetTrackingCut(fpTrackingCutCmd->GetNewBoolValue(newValue));
    } else {
      G4cout << "### DetectorMessenger Warning: addIonsTrackingCut is only supported for PhysicsList (ice)."
             << G4endl;
    }
  }

  if (command == fDensityCmd)
   {
     G4double dens;
     G4String name, unt;
     std::istringstream is(newValue);
     is >> name >> dens >> unt;
     dens *= G4UIcommand::ValueOf(unt);
     fpDetector->MaterialWithDensity(name,dens);
     fpDetector->SetMaterial(name);
   }

  if (command == fSizeCmd)
    fpDetector->SetSize(fSizeCmd->GetNewDoubleValue(newValue));

  if (command == fIceSizeCmd) {
    const auto vec = fIceSizeCmd->GetNew3VectorValue(newValue);
    fpDetector->SetIceSize(vec.x(), vec.y(), vec.z());
  }

  if (command == fMaxThetaCmd) {
    const G4bool useGeometryLimitedTheta = fMaxThetaCmd->GetNewBoolValue(newValue);
    G4double thetaDeg = 90.0;

    if (useGeometryLimitedTheta) {
      auto* gpsData = G4GeneralParticleSourceData::Instance();
      auto* source = gpsData ? gpsData->GetCurrentSource() : nullptr;
      auto* posDist = source ? source->GetPosDist() : nullptr;

      if (!posDist) {
        G4cout << "DetectorMessenger: /dna/test/setMaxTheta true requires GPS "
               << "with valid position distribution." << G4endl;
        return;
      }

      const G4ThreeVector sourcePos = posDist->GetCentreCoords();

      // Current implementation assumes the incoming surface normal is +z
      // (the standard slab setup used in this project).
      const G4double zDistToTop = -sourcePos.z();  // slab starts at z=0
      const G4double marginX = 0.5 * fpDetector->GetIceSizeX() - std::abs(sourcePos.x());
      const G4double marginY = 0.5 * fpDetector->GetIceSizeY() - std::abs(sourcePos.y());
      const G4double minMargin = std::min(marginX, marginY);

      if (zDistToTop > 0.0 && minMargin > 0.0) {
        thetaDeg = std::atan(minMargin / zDistToTop) / deg;
        thetaDeg = std::max(0.0, std::min(89.999999, thetaDeg));
      } else if (minMargin <= 0.0) {
        thetaDeg = 0.0;
      } else {
        G4cout << "DetectorMessenger: /dna/test/setMaxTheta true expects source z < 0 "
               << "for slab-at-z=0 geometry. Using 90 deg." << G4endl;
      }
    }

    std::ostringstream cmd;
    cmd << std::fixed << std::setprecision(6)
        << "/gps/ang/maxtheta " << thetaDeg << " deg";
    auto* ui = G4UImanager::GetUIpointer();
    const G4int status = ui->ApplyCommand(cmd.str().c_str());
    if (status != 0) {
      G4cout << "DetectorMessenger: failed to apply command: " << cmd.str()
             << " (status=" << status << ")" << G4endl;
    } else {
      G4cout << "DetectorMessenger: "
             << (useGeometryLimitedTheta ? "setMaxTheta true" : "setMaxTheta false")
             << " -> /gps/ang/maxtheta " << thetaDeg << " deg" << G4endl;
    }
  }

  if (command == fLogModeCmd) {
    RunAction::SetLogMode(newValue);
  }
}
