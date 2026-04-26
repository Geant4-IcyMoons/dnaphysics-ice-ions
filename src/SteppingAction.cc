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
/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class

#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

#include "G4Box.hh"
#include "G4Alpha.hh"
#include "G4AnalysisManager.hh"
#include "G4DNAGenericIonsManager.hh"
#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4String.hh"
#include "G4Gamma.hh"
#include "G4Proton.hh"
#include "G4SteppingManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ProcessManager.hh"
#include "G4VProcess.hh"

#include "G4VEmProcess.hh"
#include "G4EmCalculator.hh"
#include "G4DNAMolecularMaterial.hh"
#include "G4Material.hh"
#include "G4DNAEmfietzoglou_iceIonisationModel.hh"
#include "G4DNAEmfietzoglou_iceExcitationModel.hh"
#include "G4DNAEmfietzoglouIonisationModelTracked.hh"
#include "G4DNAEmfietzoglouExcitationModelTracked.hh"
#include "G4DNABornIonisationModel1Tracked.hh"
#include "G4DNABornExcitationModel1Tracked.hh"
#include "G4DNAMichaudExcitationModel.hh"
#include "G4DNASancheExcitationModel.hh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <set>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(EventAction* eventAction)
  : G4UserSteppingAction(), fEventAction(eventAction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Static storage for per-step logs
static std::vector<SteppingAction::StepRecord> g_stepLogs;
static bool g_logEnabled = false;
static std::set<std::string> g_observedModels;
static std::mutex g_observedMutex;

std::vector<SteppingAction::StepRecord>& SteppingAction::Logs() { return g_stepLogs; }
void SteppingAction::ClearLogs() { g_stepLogs.clear(); }
void SteppingAction::SetLoggingEnabled(bool enabled) { g_logEnabled = enabled; }
bool SteppingAction::IsLoggingEnabled() { return g_logEnabled; }
void SteppingAction::ClearObservedModels()
{
  std::lock_guard<std::mutex> lock(g_observedMutex);
  g_observedModels.clear();
}
std::vector<std::string> SteppingAction::ObservedModels()
{
  std::lock_guard<std::mutex> lock(g_observedMutex);
  return std::vector<std::string>(g_observedModels.begin(), g_observedModels.end());
}

namespace {
void RecordObservedModel(const std::string& value)
{
  std::lock_guard<std::mutex> lock(g_observedMutex);
  g_observedModels.insert(value);
}
}

namespace {
bool ReadEnvFlag(const char* name, bool defaultValue)
{
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return defaultValue;
  }
  return std::strcmp(env, "0") != 0;
}

}

// Thread-local cached state for high-energy fallback toggling.
// Reset to -1 at the start of every new track so the first step always
// evaluates correctly, even if the previous track ended in the same state.
static thread_local int g_fallbackLastState = -1;

void SteppingAction::ResetHighEnergyFallbackState()
{
  g_fallbackLastState = -1;
}

void SteppingAction::SetHighEnergyFallbackActive(const G4Track* track)
{
  if (!track) return;
  if (track->GetDefinition() != G4Electron::ElectronDefinition()) return;
  constexpr G4double kHighMin = 10. * MeV;
  const bool enable = (track->GetKineticEnergy() >= kHighMin);
  const int state = enable ? 1 : 0;
  if (state == g_fallbackLastState) return;
  g_fallbackLastState = state;

  auto* pm = track->GetDefinition()->GetProcessManager();
  if (!pm) return;
  auto* plist = pm->GetProcessList();
  if (!plist) return;
  const size_t n = plist->size();
  for (size_t i = 0; i < n; ++i) {
    auto* proc = (*plist)[i];
    if (!proc) continue;
    const auto& name = proc->GetProcessName();
    if (name == "msc" || name == "eIoni" || name == "eBrem" ||
        name == "CoulombScat" || name == "CoulombScattering") {
      pm->SetProcessActivation(proc, enable);
    }
  }
}

namespace {
bool IsIceToWorldEscape(const G4StepPoint* preStep, const G4StepPoint* postStep)
{
  if (!preStep || !postStep) return false;
  if (postStep->GetStepStatus() != fGeomBoundary) return false;
  const auto* postProc = postStep->GetProcessDefinedStep();
  if (!postProc || postProc->GetProcessName() != "Transportation") return false;
  const auto* prePV = preStep->GetPhysicalVolume();
  const auto* postPV = postStep->GetPhysicalVolume();
  if (!prePV || !postPV) return false;
  return (prePV->GetName() == "Ice" && postPV->GetName() == "World");
}

G4int ClassifyEscapeFace(const G4StepPoint* preStep, const G4StepPoint* postStep)
{
  if (!preStep || !postStep) {
    return EventAction::kEscapeUnknown;
  }

  const auto* prePV = preStep->GetPhysicalVolume();
  if (!prePV || !prePV->GetLogicalVolume()) {
    return EventAction::kEscapeUnknown;
  }

  const auto* box = dynamic_cast<const G4Box*>(prePV->GetLogicalVolume()->GetSolid());
  if (!box) {
    return EventAction::kEscapeUnknown;
  }

  const auto center = prePV->GetObjectTranslation();
  const auto local = postStep->GetPosition() - center;
  const G4double dx = std::abs(std::abs(local.x()) - box->GetXHalfLength());
  const G4double dy = std::abs(std::abs(local.y()) - box->GetYHalfLength());
  const G4double dz = std::abs(std::abs(local.z()) - box->GetZHalfLength());

  if (dz <= dx && dz <= dy) {
    return (local.z() < 0.0) ? EventAction::kEscapeTop : EventAction::kEscapeBottom;
  }

  if (dx <= dz || dy <= dz) {
    return EventAction::kEscapeSide;
  }

  return EventAction::kEscapeUnknown;
}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Protection
  if (!step->GetPostStepPoint()) return;
  if (!step->GetPostStepPoint()->GetProcessDefinedStep()) return;

  //
  G4double flagParticle = -1.;
  G4double flagProcess = -1.;
  G4double x, y, z, xp, yp, zp;

// Particle identification
  G4ParticleDefinition* partDef = step->GetTrack()->GetDynamicParticle()->GetDefinition();

  if (partDef == G4Gamma::GammaDefinition()) flagParticle = 0;
  if (partDef == G4Electron::ElectronDefinition()) flagParticle = 1;
  if (partDef == G4Proton::ProtonDefinition()) flagParticle = 2;
  if (partDef == G4Alpha::AlphaDefinition()) flagParticle = 4;

  G4DNAGenericIonsManager* instance = G4DNAGenericIonsManager::Instance();
  if (partDef == instance->GetIon("hydrogen")) flagParticle = 3;
  if (partDef == instance->GetIon("alpha+")) flagParticle = 5;
  if (partDef == instance->GetIon("helium")) flagParticle = 6;

  // Toggle high-energy fallback processes based on current kinetic energy
  SteppingAction::SetHighEnergyFallbackActive(step->GetTrack());

  // Process identification
  G4StepPoint* preStep = step->GetPreStepPoint();
  G4StepPoint* postStep = step->GetPostStepPoint();
  G4int procID = postStep->GetProcessDefinedStep()->GetProcessSubType();
  const G4String& processName = postStep->GetProcessDefinedStep()->GetProcessName();
  G4String modelName;
  if (auto* emProc = dynamic_cast<const G4VEmProcess*>(postStep->GetProcessDefinedStep())) {
    auto* model = emProc->GetCurrentModel();
    if (model) {
      modelName = model->GetName();
    }
  }

  static const G4bool kKillTrackOnIceEscape =
    ReadEnvFlag("DNA_KILL_TRACK_ON_ICE_ESCAPE", true);
  if (fEventAction && preStep && preStep->GetPhysicalVolume() &&
      preStep->GetPhysicalVolume()->GetName() == "Ice") {
    fEventAction->AddDepositedEnergy(step->GetTotalEnergyDeposit());
    if (IsIceToWorldEscape(preStep, postStep)) {
      const G4int escapeFace = ClassifyEscapeFace(preStep, postStep);
      fEventAction->AddEscapedKineticEnergy(
        postStep->GetKineticEnergy(),
        static_cast<G4int>(flagParticle),
        escapeFace
      );
      // Stop escaped tracks at the Ice boundary to avoid long straight
      // trajectory segments through vacuum up to the world boundary.
      if (kKillTrackOnIceEscape) {
        auto* track = step->GetTrack();
        track->SetTrackStatus(fStopAndKill);
      }
    }
  }

  if (fEventAction &&
      partDef == G4Electron::ElectronDefinition() &&
      processName.find("G4DNAIonisation") != std::string::npos) {
    fEventAction->AddInelastic();
  }


  if (processName == "Capture") flagProcess = 1;
  // (no subType and procID exists at the moment for this process)
  // used to kill ions below tracking cut

  else if (flagParticle == 0) {
    if (procID == 12)
      flagProcess = 81;
    else if (procID == 13)
      flagProcess = 82;
    else if (procID == 14)
      flagProcess = 83;
    else if (procID == 11)
      flagProcess = 84;
  }

  else if (flagParticle == 1) {
    if (procID == 58)
      flagProcess = 10;
    else if (procID == 51)
      flagProcess = 11;
    else if (procID == 52)
      flagProcess = 12;
    else if (procID == 53)
      flagProcess = 13;
    else if (procID == 55)
      flagProcess = 14;
    else if (procID == 54)
      flagProcess = 15;
    else if (procID == 10)
      flagProcess = 110;
    else if (procID == 1)
      flagProcess = 120;
    else if (procID == 2)
      flagProcess = 130;
  }

  else if (flagParticle == 2) {
    if (procID == 51)
      flagProcess = 21;
    else if (procID == 52)
      flagProcess = 22;
    else if (procID == 53)
      flagProcess = 23;
    else if (procID == 56)
      flagProcess = 24;
    else if (procID == 10)
      flagProcess = 210;
    else if (procID == 1)
      flagProcess = 220;
    else if (procID == 2)
      flagProcess = 230;
    else if (procID == 8)
      flagProcess = 240;
  }

  else if (flagParticle == 3) {
    if (procID == 51)
      flagProcess = 31;
    else if (procID == 52)
      flagProcess = 32;
    else if (procID == 53)
      flagProcess = 33;
    else if (procID == 57)
      flagProcess = 35;
  }

  else if (flagParticle == 4) {
    if (procID == 51)
      flagProcess = 41;
    else if (procID == 52)
      flagProcess = 42;
    else if (procID == 53)
      flagProcess = 43;
    else if (procID == 56)
      flagProcess = 44;
    else if (procID == 10)
      flagProcess = 410;
    else if (procID == 1)
      flagProcess = 420;
    else if (procID == 2)
      flagProcess = 430;
    else if (procID == 8)
      flagProcess = 440;
  }

  else if (flagParticle == 5) {
    if (procID == 51)
      flagProcess = 51;
    else if (procID == 52)
      flagProcess = 52;
    else if (procID == 53)
      flagProcess = 53;
    else if (procID == 56)
      flagProcess = 54;
    else if (procID == 57)
      flagProcess = 55;
    else if (procID == 10)
      flagProcess = 510;
    else if (procID == 1)
      flagProcess = 520;
    else if (procID == 2)
      flagProcess = 530;
    else if (procID == 8)
      flagProcess = 540;
  }

  else if (flagParticle == 6) {
    if (procID == 51)
      flagProcess = 61;
    else if (procID == 52)
      flagProcess = 62;
    else if (procID == 53)
      flagProcess = 63;
    else if (procID == 57)
      flagProcess = 65;
  }

  else if (processName == "GenericIon_G4DNAIonisation")
    flagProcess = 73;
  else if (processName == "msc")
    flagProcess = 710;
  else if (processName == "CoulombScat")
    flagProcess = 720;
  else if (processName.find("ElectronTrappingKill") != std::string::npos)
    flagProcess = 10;
  else if (processName == "ionIoni")
    flagProcess = 730;
  else if (processName == "nuclearStopping")
    flagProcess = 740;
  // (for all GenericIons)

  // Alternatively, using process names

  /*
  else if (processName=="e-_G4DNAElectronSolvation")    flagProcess =10;
  else if (processName=="e-_G4DNAElastic")              flagProcess =11;
  else if (processName=="e-_G4DNAExcitation")           flagProcess =12;
  else if (processName=="e-_G4DNAIonisation")           flagProcess =13;
  else if (processName=="e-_G4DNAAttachment")           flagProcess =14;
  else if (processName=="e-_G4DNAVibExcitation")        flagProcess =15;

  else if (processName=="proton_G4DNAElastic")          flagProcess =21;
  else if (processName=="proton_G4DNAExcitation")       flagProcess =22;
  else if (processName=="proton_G4DNAIonisation")       flagProcess =23;
  else if (processName=="proton_G4DNAChargeDecrease")   flagProcess =24;

  else if (processName=="hydrogen_G4DNAElastic")        flagProcess =31;
  else if (processName=="hydrogen_G4DNAExcitation")     flagProcess =32;
  else if (processName=="hydrogen_G4DNAIonisation")     flagProcess =33;
  else if (processName=="hydrogen_G4DNAChargeIncrease") flagProcess =35;

  else if (processName=="alpha_G4DNAElastic")           flagProcess =41;
  else if (processName=="alpha_G4DNAExcitation")        flagProcess =42;
  else if (processName=="alpha_G4DNAIonisation")        flagProcess =43;
  else if (processName=="alpha_G4DNAChargeDecrease")    flagProcess =44;

  else if (processName=="alpha+_G4DNAElastic")          flagProcess =51;
  else if (processName=="alpha+_G4DNAExcitation")       flagProcess =52;
  else if (processName=="alpha+_G4DNAIonisation")       flagProcess =53;
  else if (processName=="alpha+_G4DNAChargeDecrease")   flagProcess =54;
  else if (processName=="alpha+_G4DNAChargeIncrease")   flagProcess =55;

  else if (processName=="helium_G4DNAElastic")          flagProcess =61;
  else if (processName=="helium_G4DNAExcitation")       flagProcess =62;
  else if (processName=="helium_G4DNAIonisation")       flagProcess =63;
  else if (processName=="helium_G4DNAChargeIncrease")   flagProcess =65;

  else if (processName=="GenericIon_G4DNAIonisation")   flagProcess =73;

  */
  if (processName != "Transportation") {
    x = preStep->GetPosition().x() / nanometer;
    y = preStep->GetPosition().y() / nanometer;
    z = preStep->GetPosition().z() / nanometer;

    xp = postStep->GetPosition().x() / nanometer;
    yp = postStep->GetPosition().y() / nanometer;
    zp = postStep->GetPosition().z() / nanometer;

    // get analysis manager

    G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
    const G4int eventId =
      G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    const G4double totalEnergyDepositForLog = step->GetTotalEnergyDeposit() / eV;
    if (RunAction::IsMinimalLogMode()) {
      analysisManager->FillNtupleDColumn(0, 0, flagParticle);
      analysisManager->FillNtupleDColumn(0, 1, xp);
      analysisManager->FillNtupleDColumn(0, 2, yp);
      analysisManager->FillNtupleDColumn(0, 3, zp);
      analysisManager->FillNtupleDColumn(0, 4, totalEnergyDepositForLog);
      analysisManager->FillNtupleDColumn(0, 5, preStep->GetKineticEnergy() / eV);
      analysisManager->FillNtupleIColumn(0, 6, eventId);
      analysisManager->FillNtupleIColumn(0, 7, step->GetTrack()->GetParentID());
      analysisManager->AddNtupleRow(0);
      return;
    }

    // fill ntuple
    analysisManager->FillNtupleDColumn(0, flagParticle);
    analysisManager->FillNtupleDColumn(1, flagProcess);
    analysisManager->FillNtupleDColumn(2, xp);
    analysisManager->FillNtupleDColumn(3, yp);
    analysisManager->FillNtupleDColumn(4, zp);
    analysisManager->FillNtupleDColumn(5, totalEnergyDepositForLog);

    analysisManager->FillNtupleDColumn(
      6, std::sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp) + (z - zp) * (z - zp)));

    analysisManager->FillNtupleDColumn(
      7, (preStep->GetKineticEnergy() - postStep->GetKineticEnergy()) / eV);

    analysisManager->FillNtupleDColumn(8, preStep->GetKineticEnergy() / eV);

    analysisManager->FillNtupleDColumn(9, preStep->GetMomentumDirection()
                                            * postStep->GetMomentumDirection());

    analysisManager->FillNtupleIColumn(10, eventId);

    analysisManager->FillNtupleIColumn(11, step->GetTrack()->GetTrackID());

    analysisManager->FillNtupleIColumn(12, step->GetTrack()->GetParentID());

    analysisManager->FillNtupleIColumn(13, step->GetTrack()->GetCurrentStepNumber());

    if (flagProcess >= 0.) {
      std::ostringstream observed;
      observed << "flagProcess=" << static_cast<int>(flagProcess)
               << ";process=" << processName;
      if (!modelName.empty()) {
        observed << ";model=" << modelName;
      }
      RecordObservedModel(observed.str());
    }

    // Compute macroscopic cross section for the actual process (if available)
    // G4EmCalculator keeps mutable state and must not be shared across worker threads.
    static thread_local G4EmCalculator* emCal = nullptr;
    if (!emCal) {
      // Avoid destructor-order crashes at shutdown by keeping this alive.
      emCal = new G4EmCalculator();
    }
    G4double sigmaPerVol = emCal->ComputeCrossSectionPerVolume(
      preStep->GetKineticEnergy(),
      step->GetTrack()->GetParticleDefinition(),
      processName,
      preStep->GetMaterial());

    // Convert to microscopic area by dividing by molecular number density
    // Default to -1 if density is not available
    G4double sigma_area_cm2 = -1.0;
    if (sigmaPerVol >= 0.) {
      auto* mat = preStep->GetMaterial();
      auto* table = G4DNAMolecularMaterial::Instance()->GetNumMolPerVolTableFor(mat);
      if (table != nullptr) {
        G4double n_per_mm3 = (*table)[mat->GetIndex()]; // 1/mm^3
        if (n_per_mm3 > 0.) {
          G4double sigma_area_mm2 = sigmaPerVol / n_per_mm3; // mm^2
          sigma_area_cm2 = sigma_area_mm2 / (cm*cm);        // cm^2
        }
      }
    }

    // Append to post-run logs
    SteppingAction::StepRecord rec;
    rec.stepNo = step->GetTrack()->GetCurrentStepNumber();
    rec.kinE_eV = preStep->GetKineticEnergy() / eV;
    rec.process = processName;
    rec.model = modelName;
    // Derive a channel label for logging/printing:
    // - Vibrational excitation: use vib_<index> when available (or nearest mode).
    // - All other processes: use a descriptive category (elastic, excitation, ionisation, attachment, ...).
    int chanIdx = -1;
    G4double chanMicroXS_cm2 = -1.0;
    if (processName.find("Vib") != std::string::npos) {
      if (modelName.find("Sanche") != std::string::npos) {
        chanIdx = G4DNASancheExcitationModel::GetLastChannelIndex();
        chanMicroXS_cm2 = G4DNASancheExcitationModel::GetLastPartialSigma_cm2();
      } else {
        // Prefer exact info exposed by the active vib model (Michaud)
        chanIdx = G4DNAMichaudExcitationModel::GetLastChannelIndex();
        chanMicroXS_cm2 = G4DNAMichaudExcitationModel::GetLastPartialSigma_cm2();
      }
      if (chanIdx >= 0) {
        std::ostringstream ch;
        ch << "vib_" << chanIdx;
        rec.channel = ch.str();
      } else {
        // Fallback: approximate channel by nearest dE to known centers
        const G4double dE_eV = (preStep->GetKineticEnergy() - postStep->GetKineticEnergy())/eV;
        static const G4double omega_michaud[8] = {0.024, 0.061, 0.092, 0.205, 0.417, 0.460, 0.510, 0.834};
        static const G4double omega_sanche[9]  = {0.010, 0.024, 0.061, 0.092, 0.204, 0.417, 0.460, 0.500, 0.835};
        const bool isSanche = (modelName.find("Sanche") != std::string::npos);
        const G4double* omega = isSanche ? omega_sanche : omega_michaud;
        const int nOmega = isSanche ? 9 : 8;
        int best = -1; G4double bestDiff = DBL_MAX;
        for (int i=0;i<nOmega;++i) {
          const G4double diff = std::abs(dE_eV - omega[i]);
          if (diff < bestDiff) { bestDiff = diff; best = i; }
        }
        chanIdx = best;
        if (best >= 0) {
          std::ostringstream ch;
          ch << "vib_" << best;
          rec.channel = ch.str();
        }
      }
    } else if (processName.find("Excitation") != std::string::npos) {
      const bool isIceEmfi = (modelName.find("Emfietzoglou_ice") != std::string::npos);
      const bool isWaterEmfi = (modelName.find("EmfietzoglouExcitationModel") != std::string::npos);
      const bool isBorn = (modelName.find("BornExcitationModel") != std::string::npos);
      int levelIdx = -1;
      if (isIceEmfi) {
        levelIdx = G4DNAEmfietzoglou_iceExcitationModel::GetLastExcitationIndex();
        chanMicroXS_cm2 = G4DNAEmfietzoglou_iceExcitationModel::GetLastPartialSigma_cm2();
      } else if (isWaterEmfi) {
        levelIdx = G4DNAEmfietzoglouExcitationModelTracked::GetLastExcitationIndex();
        chanMicroXS_cm2 = G4DNAEmfietzoglouExcitationModelTracked::GetLastPartialSigma_cm2();
      } else if (isBorn) {
        levelIdx = G4DNABornExcitationModel1Tracked::GetLastExcitationIndex();
        chanMicroXS_cm2 = G4DNABornExcitationModel1Tracked::GetLastPartialSigma_cm2();
      }
      if (levelIdx >= 0) {
        chanIdx = levelIdx;
        std::ostringstream ch;
        ch << "exc_" << levelIdx;
        rec.channel = ch.str();
      } else {
        // Fallback: infer channel from energy loss and excitation energies
        const G4double dE_eV =
            (preStep->GetKineticEnergy() - postStep->GetKineticEnergy()) / eV;
        static const G4double excE[5] = {8.22, 10.00, 11.24, 12.61, 13.77};
        int best = -1;
        G4double bestDiff = DBL_MAX;
        for (int i = 0; i < 5; ++i) {
          const G4double diff = std::abs(dE_eV - excE[i]);
          if (diff < bestDiff) {
            bestDiff = diff;
            best = i;
          }
        }
        chanIdx = best;
        if (best >= 0) {
          std::ostringstream ch;
          ch << "exc_" << best;
          rec.channel = ch.str();
        } else {
          rec.channel = "excitation";
        }
      }
    } else if (processName.find("Ionis") != std::string::npos ||
               processName.find("Ioniz") != std::string::npos) {
      const bool isIceEmfi = (modelName.find("Emfietzoglou_ice") != std::string::npos);
      const bool isWaterEmfi = (modelName.find("EmfietzoglouIonisationModel") != std::string::npos);
      const bool isBorn = (modelName.find("BornIonisationModel") != std::string::npos);
      int shellIdx = -1;
      if (isIceEmfi) {
        shellIdx = G4DNAEmfietzoglou_iceIonisationModel::GetLastShellIndex();
        chanMicroXS_cm2 = G4DNAEmfietzoglou_iceIonisationModel::GetLastPartialSigma_cm2();
      } else if (isWaterEmfi) {
        shellIdx = G4DNAEmfietzoglouIonisationModelTracked::GetLastShellIndex();
        chanMicroXS_cm2 = G4DNAEmfietzoglouIonisationModelTracked::GetLastPartialSigma_cm2();
      } else if (isBorn) {
        shellIdx = G4DNABornIonisationModel1Tracked::GetLastShellIndex();
        chanMicroXS_cm2 = G4DNABornIonisationModel1Tracked::GetLastPartialSigma_cm2();
      }
      if (shellIdx >= 0) {
        chanIdx = shellIdx;
        std::ostringstream ch;
        ch << "ion_" << shellIdx;
        rec.channel = ch.str();
      } else {
        // Fallback: infer channel from energy loss and binding energies
        const G4double dE_eV =
            (preStep->GetKineticEnergy() - postStep->GetKineticEnergy()) / eV;
        static const G4double bindingE[5] = {10.0, 13.0, 17.0, 32.2, 539.7};
        int best = -1;
        G4double bestDiff = DBL_MAX;
        for (int i = 0; i < 5; ++i) {
          if (dE_eV >= bindingE[i]) {
            G4double diff = dE_eV - bindingE[i];
            if (diff < bestDiff && diff < 1000.0) {
              bestDiff = diff;
              best = i;
            }
          }
        }
        chanIdx = best;
        if (best >= 0) {
          std::ostringstream ch;
          ch << "ion_" << best;
          rec.channel = ch.str();
        } else {
          rec.channel = "ionisation";
        }
      }
    } else {
      // Map processName to a human-friendly category label
      std::string label;
      if (processName.find("Elastic") != std::string::npos)            label = "elastic";
      else if (processName.find("Attach") != std::string::npos)         label = "attachment";
      else if (processName.find("Excitation") != std::string::npos)     label = "excitation";
      else if (processName.find("Charge") != std::string::npos)         label = "charge";
      else if (processName.find("msc") != std::string::npos)            label = "msc";
      else if (processName.find("Coulomb") != std::string::npos)        label = "coulomb";
      else                                                               label = processName;
      rec.channel = label;
    }
    if (chanIdx < 0) {
      if (processName.find("Elastic") != std::string::npos ||
          processName.find("Attach") != std::string::npos) {
        chanIdx = 0;
      }
    }
    // Use per-channel microscopic XS for vib when available; otherwise use per-volume XS converted to area
    if (processName.find("Vib") != std::string::npos && chanMicroXS_cm2 > 0.) {
      rec.sigma_area_cm2 = chanMicroXS_cm2;
    } else {
      rec.sigma_area_cm2 = sigma_area_cm2;
    }
    if (g_logEnabled) {
      g_stepLogs.push_back(rec);
    }

    // Keep ntuple column 14 to carry the per-step sigma (1/cm)
    analysisManager->FillNtupleDColumn(14, sigmaPerVol);
    // New: record channel index (if identified), otherwise -1
    analysisManager->FillNtupleIColumn(15, chanIdx);
    // New: per-channel microscopic XS in cm^2 if available (else -1)
    analysisManager->FillNtupleDColumn(16, chanMicroXS_cm2);
    if (ReadEnvFlag("DNA_NTUPLE_STRINGS", true)) {
      analysisManager->FillNtupleSColumn(17, processName);
      analysisManager->FillNtupleSColumn(18, modelName);
    }

    analysisManager->AddNtupleRow();
  }
}
