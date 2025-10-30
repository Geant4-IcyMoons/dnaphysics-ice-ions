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
// ---------------------------------------------------------------------
//  G4DNABrownianTransportation_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  Implements Brownian motion/transport for molecules in ice, using
//  temperature-dependent diffusion. Differs from the original by using
//  Arrhenius coefficients and ice-specific transport logic.
// ---------------------------------------------------------------------
//
// Author: Mathieu Karamitros (kara (AT) cenbg . in2p3 . fr) 
//
// WARNING : This class is released as a prototype.
// It might strongly evolve or even disapear in the next releases.
//
// History:
// -----------
// 10 Oct 2011 M.Karamitros created
//
// -------------------------------------------------------------------

/// \brief { The transportation method implemented is the one from
///         Ermak-McCammon : J. Chem. Phys. 69, 1352 (1978)}

#include "G4DNABrownianTransportation_ice.hh"

#include "G4DNAMolecularMaterial.hh"
#include "G4ITNavigator.hh"
#include "G4MolecularConfiguration.hh"
#include "G4ITSafetyHelper.hh" // Not used yet
#include "G4LowEnergyEmProcessSubType.hh"
#include "G4Molecule.hh"
#include "G4NistManager.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4RandomDirection.hh"
#include "G4SafetyHelper.hh"
#include "G4SystemOfUnits.hh"
#include "G4TrackingInformation.hh"
#include "G4TransportationManager.hh"
#include "G4UnitsTable.hh"
#include "G4VUserBrownianAction.hh"
#include "Randomize.hh"

// Ice-specific includes
#include "G4OH.hh"
#include "G4Hydrogen.hh"
#include "G4MoleculeTable.hh"
#include "ArrheniusConstants_ice.hh"


#include <CLHEP/Random/Stat.h>
#include <G4Scheduler.hh>
#include <sstream>

#include <memory>
using namespace std;

#ifndef State
#define State(theXInfo) (GetState<G4ITBrownianState>()->theXInfo)
#endif

//#ifndef State
//#define State(theXInfo) (fTransportationState->theXInfo)
//#endif

//#define USE_COLOR 1

#ifdef USE_COLOR
#define RED  "\033[0;31m"
#define LIGHT_RED  "\33[1;31m"
#define GREEN "\033[32;40m"
#define GREEN_ON_BLUE "\033[1;32;44m"
#define RESET_COLOR "\033[0m"
#else
#define RED  ""
#define LIGHT_RED  ""
#define GREEN ""
#define GREEN_ON_BLUE ""
#define RESET_COLOR ""
#endif

//#define DEBUG_MEM 1

#ifdef DEBUG_MEM
#include "G4MemStat.hh"
using namespace G4MemStat;
using G4MemStat::MemStat;
#endif

static G4double InvErf(G4double x)
{
  return CLHEP::HepStat::inverseErf(x);
}

static G4double InvErfc(G4double x)
{
  return CLHEP::HepStat::inverseErf(1. - x);
}

//static double Erf(double x)
//{
//  return CLHEP::HepStat::erf(x);
//}

static G4double Erfc(G4double x)
{
  return 1 - CLHEP::HepStat::erf(1. - x);
}

#ifndef State
#define State(theXInfo) (GetState<G4ITTransportationState>()->theXInfo)
#endif

// Arrhenius parameters now centralized in ArrheniusConstants_ice.hh

// Global temperature profile data - to be set by user application
static std::vector<G4double> g_zRanges;
static std::vector<G4double> g_zTemperatures;
static G4bool g_temperatureProfileSet = false;

// Function to set temperature profile from user application
void SetIceTemperatureProfile(const std::vector<G4double>& zRanges, const std::vector<G4double>& zTemperatures)
{
  g_zRanges = zRanges;
  g_zTemperatures = zTemperatures;
  g_temperatureProfileSet = true;
  
  G4cout << "Ice temperature profile set:" << G4endl;
  G4cout << "  Z ranges: ";
  for (size_t i = 0; i < g_zRanges.size(); ++i) {
    G4cout << g_zRanges[i];
    if (i < g_zRanges.size() - 1) G4cout << " ";
  }
  G4cout << " mm" << G4endl;
  G4cout << "  Temperatures: ";
  for (size_t i = 0; i < g_zTemperatures.size(); ++i) {
    G4cout << g_zTemperatures[i];
    if (i < g_zTemperatures.size() - 1) G4cout << " ";
  }
  G4cout << " K" << G4endl;
  
  // Verify Arrhenius diffusion coefficients for each layer
  G4cout << "\nArrhenius diffusion coefficient verification:" << G4endl;
  G4cout << "OH parameters: D0=" << ArrheniusConstants_ice::D0_OH/(m2/s) << " m2/s, Ea=" << ArrheniusConstants_ice::Ea_OH/eV << " eV" << G4endl;
  G4cout << "H parameters:  D0=" << ArrheniusConstants_ice::D0_H/(m2/s) << " m2/s, Ea=" << ArrheniusConstants_ice::Ea_H/eV << " eV" << G4endl;
  
  for (size_t i = 0; i < g_zTemperatures.size(); ++i) {
    G4double T = g_zTemperatures[i] * kelvin;
    G4double D_OH = ArrheniusConstants_ice::CalculateArrheniusDiffusion(ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, T);
    G4double D_H = ArrheniusConstants_ice::CalculateArrheniusDiffusion(ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, T);
    
    G4cout << "  Layer " << (i+1) << " (T=" << g_zTemperatures[i] << "K): "
           << "D_OH=" << D_OH/(m2/s) << " m2/s, "
           << "D_H=" << D_H/(m2/s) << " m2/s" << G4endl;
  }
  G4cout << G4endl;
}

// Helper function to get temperature from position using discrete depth intervals
G4double GetIceTemperatureAtPosition(const G4ThreeVector& position)
{
  G4double z_position = position.z();
  G4double depth_mm = z_position / mm;
  
  // Check if temperature profile has been set
  if (!g_temperatureProfileSet) {
    G4Exception("GetTemperatureAtPosition", "IceTransport001", FatalException,
                "Temperature profile not set! Call SetIceTemperatureProfile() before simulation.");
  }
  
  // Validate that we have temperature data
  if (g_zRanges.empty() || g_zTemperatures.empty()) {
    G4Exception("GetTemperatureAtPosition", "IceTransport002", FatalException,
                "Empty temperature profile! Check SetIceTemperatureProfile() arguments.");
  }
  
  if (g_zTemperatures.size() != g_zRanges.size() - 1) {
    G4Exception("GetTemperatureAtPosition", "IceTransport003", FatalException,
                "Temperature array size mismatch with depth ranges! Check SetIceTemperatureProfile() arguments.");
  }
  
  // Find the appropriate temperature based on depth intervals
  for (size_t i = 0; i < g_zRanges.size() - 1; ++i) {
    if (depth_mm >= g_zRanges[i] && depth_mm < g_zRanges[i + 1]) {
      return g_zTemperatures[i] * kelvin;
    }
  }
  
  // If beyond the last range, use the last temperature
  if (depth_mm >= g_zRanges.back()) {
    return g_zTemperatures.back() * kelvin;
  }
  
  // If before the first range, use the first temperature
  return g_zTemperatures[0] * kelvin;
}

// Helper function to get position-dependent diffusion coefficient for OH
static G4double GetOHDiffusionCoefficient(const G4Track& track)
{
  G4Molecule* molecule = GetMolecule(track);
  
  // Check if this is an OH molecule
  if (molecule && molecule->GetDefinition() == G4OH::Definition()) {
    G4ThreeVector position = track.GetPosition();
    G4double temperature = GetIceTemperatureAtPosition(position);
    
    // Calculate position-dependent diffusion coefficient using Arrhenius equation
    G4double diffusionCoeff = ArrheniusConstants_ice::CalculateArrheniusDiffusion(ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, temperature);
    
    // Print movement debug info with track ID to identify specific molecules (high precision)
    // G4cout << "DEBUG: OH molecule (ID=" << track.GetTrackID() << ") at z = " << std::setprecision(10) << position.z()/mm << " mm, T = " << temperature/kelvin 
    //        << " K, D = " << diffusionCoeff/(m2/s) << " m²/s, time = " << track.GetGlobalTime()/picosecond << " ps" << G4endl;
    
    // Return the calculated diffusion coefficient
    return diffusionCoeff;
  }
  
  // For non-OH molecules, use the default diffusion coefficient
  return molecule->GetDiffusionCoefficient();
}

// Helper function to get position-dependent diffusion coefficient for hydrogen
static G4double GetHydrogenDiffusionCoefficient(const G4Track& track)
{
  G4Molecule* molecule = GetMolecule(track);
  
  // Check if this is a hydrogen molecule
  if (molecule && molecule->GetDefinition() == G4Hydrogen::Definition()) {
    G4ThreeVector position = track.GetPosition();
    G4double temperature = GetIceTemperatureAtPosition(position);
    
    // Calculate position-dependent diffusion coefficient using Arrhenius equation
    G4double diffusionCoeff = ArrheniusConstants_ice::CalculateArrheniusDiffusion(ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, temperature);
    
    // Print movement debug info with track ID to identify specific molecules (high precision)
    // G4cout << "DEBUG: H molecule (ID=" << track.GetTrackID() << ") at z = " << std::setprecision(10) << position.z()/mm << " mm, T = " << temperature/kelvin 
    //        << " K, D = " << diffusionCoeff/(m2/s) << " m²/s, time = " << track.GetGlobalTime()/picosecond << " ps" << G4endl;
    
    // Return the calculated diffusion coefficient
    return diffusionCoeff;
  }
  
  // For non-hydrogen molecules, use the default diffusion coefficient
  return molecule->GetDiffusionCoefficient();
}

G4DNABrownianTransportation_ice::G4DNABrownianTransportation_ice(const G4String& aName,
                                                         G4int verbosity) :
    G4ITTransportation(aName, verbosity)
{

  fVerboseLevel = 0;

  fpState = std::make_shared<G4ITBrownianState>();

  //ctor
  SetProcessSubType(fLowEnergyBrownianTransportation);

  fNistWater = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  fpWaterDensity = nullptr;

  fUseMaximumTimeBeforeReachingBoundary = true;
  fUseSchedulerMinTimeSteps = false;
  fSpeedMeUp = true;

  fInternalMinTimeStep = 1*picosecond;
  fpBrownianAction = nullptr;
  fpUserBrownianAction = nullptr;
}

G4DNABrownianTransportation_ice::~G4DNABrownianTransportation_ice()
{
  delete fpUserBrownianAction;
}

G4DNABrownianTransportation_ice::G4ITBrownianState::G4ITBrownianState()    
{
  fPathLengthWasCorrected = false;
  fTimeStepReachedLimit = false;
  fComputeLastPosition = false;
  fRandomNumber = -1;
}

void G4DNABrownianTransportation_ice::StartTracking(G4Track* track)
{
  fpState = std::make_shared<G4ITBrownianState>();
//	G4cout << "G4DNABrownianTransportation_ice::StartTracking : "
//  "Initialised track State" << G4endl;

  // Hydrogen molecule tracking will be handled at application level
  // in ChemSteppingAction instead of here

  SetInstantiateProcessState(false);
  G4ITTransportation::StartTracking(track);
}

void G4DNABrownianTransportation_ice::BuildPhysicsTable(const G4ParticleDefinition& particle)
{
  if(verboseLevel > 0)
  {
    G4cout << G4endl<< GetProcessName() << ":   for  "
    << setw(24) << particle.GetParticleName()
    << "\tSubType= " << GetProcessSubType() << G4endl;
  }
  // Initialize water density pointer
  fpWaterDensity = G4DNAMolecularMaterial::Instance()->
  GetDensityTableFor(G4Material::GetMaterial("G4_WATER"));

  fpSafetyHelper->InitialiseHelper();
  G4ITTransportation::BuildPhysicsTable(particle);
}

void G4DNABrownianTransportation_ice::ComputeStep(const G4Track& track,
                                              const G4Step& step,
                                              const G4double timeStep,
                                              G4double& spaceStep)
{
  // G4cout << "G4ITBrownianTransportation::ComputeStep" << G4endl;

  /* If this method is called, this step
   * cannot be geometry limited.
   * In case the step is limited by the geometry,
   * this method should not be called.
   * The fTransportEndPosition calculated in
   * the method AlongStepIL should be taken
   * into account.
   * In order to do so, the flag IsLeadingStep
   * is on. Meaning : this track has the minimum
   * interaction length over all others.
   */
  if(GetIT(track)->GetTrackingInfo()->IsLeadingStep())
  {
    const G4VITProcess* ITProc = ((const G4VITProcess*) step.GetPostStepPoint()
        ->GetProcessDefinedStep());
    G4bool makeException = true;

    if((ITProc != nullptr) && ITProc->ProposesTimeStep()) makeException = false;

    if(makeException)
    {

      G4ExceptionDescription exceptionDescription;
      exceptionDescription << "ComputeStep is called while the track has"
                              "the minimum interaction time";
      exceptionDescription << " so it should not recompute a timeStep ";
      G4Exception("G4DNABrownianTransportation_ice::ComputeStep",
                  "G4DNABrownianTransportation001", FatalErrorInArgument,
                  exceptionDescription);
    }
  }

  State(fGeometryLimitedStep) = false;

  G4Molecule* molecule = GetMolecule(track);

  if(timeStep > 0)
  {
    spaceStep = DBL_MAX;

    // Ice-specific: Use position-dependent Arrhenius diffusion for OH and Hydrogen
    G4double diffCoeff;
    
    if (molecule && molecule->GetDefinition() == G4OH::Definition()) {
      // For OH molecules, use Arrhenius temperature-dependent diffusion
      diffCoeff = GetOHDiffusionCoefficient(track);
      
      // Debug: Always print OH molecule diffusion info with layer information
      G4ThreeVector position = track.GetPosition();
      G4double temperature = GetIceTemperatureAtPosition(position);
      G4double depth_mm = position.z() / mm;
      
      // Get layer information for better debugging
      G4String layer_info = "Unknown";
      
      if (!g_zRanges.empty()) {
        for (size_t i = 0; i < g_zRanges.size() - 1; ++i) {
          if (depth_mm >= g_zRanges[i] && depth_mm < g_zRanges[i + 1]) {
            std::ostringstream oss;
            oss << "Layer-" << (i + 1) << " (" << g_zRanges[i] << "-" << g_zRanges[i + 1] << "mm)";
            layer_info = oss.str();
            break;
          }
        }
        
        // Handle beyond last range
        if (depth_mm >= g_zRanges.back()) {
          std::ostringstream oss;
          oss << "Deep (>" << g_zRanges.back() << "mm)";
          layer_info = oss.str();
        }
      }
      
      // Debug: OH molecule diffusion info with Arrhenius verification
      // G4cout << "DEBUG-ARRHENIUS: " << molecule->GetName() << " at z=" << depth_mm << "mm [" << layer_info << "], T=" << temperature/kelvin 
      //        << "K, D=" << diffCoeff/(m2/s) << " m2/s (Arrhenius-macro)" << G4endl;
      
      if (fVerboseLevel > 1) {
        G4cout << "Ice transport: OH at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
               << " K, D=" << diffCoeff/(m2/s) << " m2/s" << G4endl;
      }
    } else if (molecule && molecule->GetDefinition() == G4Hydrogen::Definition()) {
      // For Hydrogen molecules, use Arrhenius temperature-dependent diffusion
      diffCoeff = GetHydrogenDiffusionCoefficient(track);
      
      // Debug: Always print Hydrogen molecule diffusion info with layer information
      G4ThreeVector position = track.GetPosition();
      G4double temperature = GetIceTemperatureAtPosition(position);
      G4double depth_mm = position.z() / mm;
      
      // Get layer information for better debugging
      G4String layer_info = "Unknown";
      
      if (!g_zRanges.empty()) {
        for (size_t i = 0; i < g_zRanges.size() - 1; ++i) {
          if (depth_mm >= g_zRanges[i] && depth_mm < g_zRanges[i + 1]) {
            std::ostringstream oss;
            oss << "Layer-" << (i + 1) << " (" << g_zRanges[i] << "-" << g_zRanges[i + 1] << "mm)";
            layer_info = oss.str();
            break;
          }
        }
        
        // Handle beyond last range
        if (depth_mm >= g_zRanges.back()) {
          std::ostringstream oss;
          oss << "Deep (>" << g_zRanges.back() << "mm)";
          layer_info = oss.str();
        }
      }
      
    //   G4cout << "DEBUG: " << molecule->GetName() << " at z=" << depth_mm << "mm [" << layer_info << "], T=" << temperature/kelvin 
    //          << "K, D=" << diffCoeff/(m2/s) << " m2/s (Arrhenius-macro)" << G4endl;
      
      // Hydrogen transport tracking handled at application level in ChemSteppingAction
      
      if (fVerboseLevel > 1) {
        G4cout << "Ice transport: H at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
               << " K, D=" << diffCoeff/(m2/s) << " m2/s" << G4endl;
      }
    } else {
      // For non-OH/non-H molecules, use standard GEANT4 diffusion
      diffCoeff = molecule->GetDiffusionCoefficient(track.GetMaterial(),
                                                   track.GetMaterial()->GetTemperature());
    }

    static G4double sqrt_2 = sqrt(2.);
    G4double sqrt_Dt = sqrt(diffCoeff*timeStep);
    G4double sqrt_2Dt = sqrt_2*sqrt_Dt;
    G4double x = G4RandGauss::shoot(0,sqrt_2Dt);
    G4double y = G4RandGauss::shoot(0,sqrt_2Dt);
    G4double z = G4RandGauss::shoot(0,sqrt_2Dt);

    if(State(fTimeStepReachedLimit)== 1)
    {
      //========================================================================
      State(fGeometryLimitedStep) = true;// important
      //========================================================================
      spaceStep = State(fEndPointDistance);
   //   G4cout << "State(fTimeStepReachedLimit)== true" << G4endl;
    }
    else
    {
      spaceStep = sqrt(x*x + y*y + z*z);

      if(spaceStep >= State(fEndPointDistance))
      {
        //G4cout << "spaceStep >= State(fEndPointDistance)" << G4endl;
        //======================================================================
        State(fGeometryLimitedStep) = true;// important
        //======================================================================
/*
        if(fSpeedMeUp)
        {
          G4cout << "SpeedMeUp" << G4endl;
        }
        else
*/
        if(!fUseSchedulerMinTimeSteps)// jump over barrier NOT used
        {
#ifdef G4VERBOSE
          if (fVerboseLevel > 1)
          {
            G4cout << GREEN_ON_BLUE
            << "G4ITBrownianTransportation::ComputeStep() : "
            << "Step was limited to boundary"
            << RESET_COLOR
            << G4endl;
          }
#endif
          //TODO
          if(State(fRandomNumber)>=0) // CDF is used
          {
            /*
             //=================================================================
             State(fGeometryLimitedStep) = true;// important
             //=================================================================
             spaceStep = State(fEndPointDistance);
             */

            //==================================================================
            // BE AWARE THAT THE TECHNIQUE USED BELOW IS A 1D APPROXIMATION
            // Cumulative density function for the 3D case is not yet
            // implemented
            //==================================================================
//            G4cout << GREEN_ON_BLUE
//                   << "G4ITBrownianTransportation::ComputeStep() : "
//                   << "A random number was selected"
//                   << RESET_COLOR
//                   << G4endl;
            G4double value = State(fRandomNumber)+(1-State(fRandomNumber))*G4UniformRand();
            G4double invErfc = InvErfc(value);
            spaceStep = invErfc*2*sqrt_Dt;

            if(State(fTimeStepReachedLimit)== 0)
            {
              //================================================================
              State(fGeometryLimitedStep) = false;// important
              //================================================================
            }
            //==================================================================
            // DEBUG
//            if(spaceStep > State(fEndPointDistance))
//            {
//              G4cout << "value = " << value << G4endl;
//              G4cout << "invErfc = " << invErfc << G4endl;
//              G4cout << "spaceStep = " << G4BestUnit(spaceStep, "Length")
//              << G4endl;
//              G4cout << "end point distance= " << G4BestUnit(State(fEndPointDistance), "Length")
//              << G4endl;
//            }
//
//            assert(spaceStep <= State(fEndPointDistance));
            //==================================================================

          }
          else if(!fUseMaximumTimeBeforeReachingBoundary) // CDF is used
          {
            G4double min_randomNumber = Erfc(State(fEndPointDistance)/2*sqrt_Dt);
            G4double value = min_randomNumber+(1-min_randomNumber)*G4UniformRand();
            G4double invErfc = InvErfc(value);
            spaceStep = invErfc*2*sqrt_Dt;
            if(spaceStep >= State(fEndPointDistance))
            {
              //================================================================
              State(fGeometryLimitedStep) = true;// important
              //================================================================
            }
            else if(State(fTimeStepReachedLimit)== 0)
            {
              //================================================================
              State(fGeometryLimitedStep) = false;// important
              //================================================================
            }
          }
          else // CDF is NOT used
          {
            //==================================================================
            State(fGeometryLimitedStep) = true;// important
            //==================================================================
            spaceStep = State(fEndPointDistance);
            //TODO

            /*
            //==================================================================
            // 1D approximation to place the brownian between its starting point
            // and the geometry boundary
            //==================================================================
            double min_randomNumber = Erfc(State(fEndPointDistance)/2*sqrt_Dt);
            double value = State(fRandomNumber)+(1-State(fRandomNumber))*G4UniformRand();
            double invErfc = InvErfc(value*G4UniformRand());
            spaceStep = invErfc*2*sqrt_Dt;
            State(fGeometryLimitedStep) = false;
            */
          }
        }

        State(fTransportEndPosition)= spaceStep*
//             step.GetPostStepPoint()->GetMomentumDirection() 
             track.GetMomentumDirection()
             + track.GetPosition();
      }
      else
      {
        //======================================================================
        State(fGeometryLimitedStep) = false;// important
        //======================================================================
        State(fTransportEndPosition)= spaceStep*step.GetPostStepPoint()->
        GetMomentumDirection() + track.GetPosition();
      }
    }
    if(fpUserBrownianAction != nullptr)
    {
      // Let the user Brownian action class decide what to do:
      G4ThreeVector nextPosition{track.GetPosition().getX() + x,
                                 track.GetPosition().getY() + y,
                                 track.GetPosition().getZ() + z};
      fpUserBrownianAction->Transport(nextPosition);
      State(fTransportEndPosition) = nextPosition;
    }
  }
  else
  {
    spaceStep = 0.;
    State(fTransportEndPosition) = track.GetPosition();
    State(fGeometryLimitedStep) = false;
  }

  State(fCandidateEndGlobalTime) = step.GetPreStepPoint()->GetGlobalTime()
      + timeStep;
  State(fEndGlobalTimeComputed) = true;

#ifdef G4VERBOSE
  //    DEBUG
  if (fVerboseLevel > 1)
  {
    G4cout << GREEN_ON_BLUE
           << "G4ITBrownianTransportation::ComputeStep() : "
           << " trackID : " << track.GetTrackID() << " : Molecule name: "
           << molecule->GetName() << G4endl
           << "Initial position:" << G4BestUnit(track.GetPosition(), "Length")
           << G4endl
           << "Initial direction:" << track.GetMomentumDirection() << G4endl
           << "Final position:" << G4BestUnit(State(fTransportEndPosition), "Length")
           << G4endl
           << "Initial magnitude:" << G4BestUnit(track.GetPosition().mag(), "Length")
           << G4endl
           << "Final magnitude:" << G4BestUnit(State(fTransportEndPosition).mag(), "Length")
           << G4endl
           << "Diffusion length : "
           << G4BestUnit(spaceStep, "Length")
           << " within time step : " << G4BestUnit(timeStep,"Time")
           << G4endl
           << "State(fTimeStepReachedLimit)= " << State(fTimeStepReachedLimit) << G4endl
           << "State(fGeometryLimitedStep)=" << State(fGeometryLimitedStep) << G4endl
           << "End point distance was: " << G4BestUnit(State(fEndPointDistance), "Length")
           << G4endl
           << RESET_COLOR
           << G4endl<< G4endl;
  }
#endif

//==============================================================================
// DEBUG
//assert(spaceStep <  State(fEndPointDistance)
//  || (spaceStep >= State(fEndPointDistance) && State(fGeometryLimitedStep)));
//assert(track.GetMomentumDirection() == State(fTransportEndMomentumDir));
//==============================================================================
}

G4VParticleChange* G4DNABrownianTransportation_ice::PostStepDoIt(const G4Track& track,
                                                             const G4Step& step)
{
  G4ITTransportation::PostStepDoIt(track, step);

#ifdef G4VERBOSE
  //    DEBUG
  if (fVerboseLevel > 1)
  {
    G4cout << GREEN_ON_BLUE << "G4ITBrownianTransportation::PostStepDoIt() :"
           << " trackID : " << track.GetTrackID() << " Molecule name: "
           << GetMolecule(track)->GetName() << G4endl;
    G4cout << "Diffusion length : "
           << G4BestUnit(step.GetStepLength(), "Length")
           <<" within time step : " << G4BestUnit(step.GetDeltaTime(),"Time")
           << "\t Current global time : "
           << G4BestUnit(track.GetGlobalTime(),"Time")
           << RESET_COLOR
           << G4endl<< G4endl;
  }
#endif
  return &fParticleChange;
}

void G4DNABrownianTransportation_ice::Diffusion(const G4Track& track)
{

#ifdef DEBUG_MEM
  MemStat mem_first = MemoryUsage();
#endif

#ifdef G4VERBOSE
  // DEBUG
  if (fVerboseLevel > 1)
  {
    G4cout << GREEN_ON_BLUE << setw(18)
           << "G4DNABrownianTransportation_ice::Diffusion :" << setw(8)
           << GetIT(track)->GetName() << "\t trackID:" << track.GetTrackID()
           << "\t" << " Global Time = "
           << G4BestUnit(track.GetGlobalTime(), "Time")
           << RESET_COLOR
           << G4endl
           << G4endl;
  }
#endif

/*
  fParticleChange.ProposePosition(State(fTransportEndPosition));
  //fParticleChange.ProposeEnergy(State(fTransportEndKineticEnergy));
  fParticleChange.SetMomentumChanged(State(fMomentumChanged));

  fParticleChange.ProposeGlobalTime(State(fCandidateEndGlobalTime));
  fParticleChange.ProposeLocalTime(State(fCandidateEndGlobalTime));
  fParticleChange.ProposeTrueStepLength(track.GetStepLength());
*/
  const G4Material* material = track.GetMaterial();

  G4double waterDensity = (*fpWaterDensity)[material->GetIndex()];

  if (waterDensity == 0.0)
  {
    if(fpBrownianAction != nullptr)
    {
      // Let the user Brownian action class decide what to do
      fpBrownianAction->Transport(track,
                                  fParticleChange);
      return;
    }
    
#ifdef G4VERBOSE
    if(fVerboseLevel != 0)
    {
      G4cout << "A track is outside water material : trackID = "
      << track.GetTrackID() << " (" << GetMolecule(track)->GetName() <<")"
      << G4endl;
      G4cout << "Local Time : " << G4BestUnit(track.GetGlobalTime(), "Time")
      << G4endl;
      G4cout << "Step Number :" << track.GetCurrentStepNumber() << G4endl;
    }
#endif
    fParticleChange.ProposeEnergy(0.);
    fParticleChange.ProposeTrackStatus(fStopAndKill);
    return;// &fParticleChange is the final returned object
  }


   #ifdef DEBUG_MEM
   MemStat mem_intermediaire = MemoryUsage();
   mem_diff = mem_intermediaire-mem_first;
   G4cout << "\t\t\t >> || MEM || In G4DNABrownianTransportation_ice::Diffusion "
   "after dealing with waterDensity for "<< track.GetTrackID()
   << ", diff is : " << mem_diff << G4endl;
   #endif

  fParticleChange.ProposeMomentumDirection(G4RandomDirection());
  State(fMomentumChanged) = true;
  fParticleChange.SetMomentumChanged(true);
   //
   #ifdef DEBUG_MEM
   mem_intermediaire = MemoryUsage();
   mem_diff = mem_intermediaire-mem_first;
   G4cout << "\t\t\t >> || MEM || In G4DNABrownianTransportation_ice::"
          "After proposing new direction to fParticleChange for "
          << track.GetTrackID() << ", diff is : " << mem_diff << G4endl;
   #endif

  return;// &fParticleChange is the final returned object
}

// NOT USED
G4double G4DNABrownianTransportation_ice::ComputeGeomLimit(const G4Track& track,
                                                       G4double& presafety,
                                                       G4double limit)
{
  G4double res = DBL_MAX;
  if(track.GetVolume() != fpSafetyHelper->GetWorldVolume())
  {
    G4TrackStateManager& trackStateMan = GetIT(track)->GetTrackingInfo()
    ->GetTrackStateManager();
    fpSafetyHelper->LoadTrackState(trackStateMan);
    res = fpSafetyHelper->CheckNextStep(
        track.GetStep()->GetPreStepPoint()->GetPosition(),
        track.GetMomentumDirection(),
        limit, presafety);
    fpSafetyHelper->ResetTrackState();
  }
  return res;
}

G4double G4DNABrownianTransportation_ice::AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                                            G4double previousStepSize,
                                                                            G4double currentMinimumStep,
                                                                            G4double& currentSafety,
                                                                            G4GPILSelection* selection)
{
#ifdef G4VERBOSE
  if(fVerboseLevel != 0)
  {
    G4cout << " G4DNABrownianTransportation_ice::AlongStepGetPhysicalInteractionLength - track ID: "
           << track.GetTrackID() << G4endl;
    G4cout << "In volume : " << track.GetVolume()->GetName()
           << " position : " << G4BestUnit(track.GetPosition() , "Length") << G4endl;
  }
#endif

  G4double geometryStepLength =
      G4ITTransportation::AlongStepGetPhysicalInteractionLength(
          track, previousStepSize, currentMinimumStep, currentSafety,
          selection);

  if(geometryStepLength==0)
  { 
//    G4cout << "geometryStepLength==0" << G4endl;
    if(State(fGeometryLimitedStep))
    {
//      G4cout << "if(State(fGeometryLimitedStep))" << G4endl;
      G4TouchableHandle newTouchable = new G4TouchableHistory;

      newTouchable->UpdateYourself(State(fCurrentTouchableHandle)->GetVolume(),
                                   State(fCurrentTouchableHandle)->GetHistory());

       fLinearNavigator->SetGeometricallyLimitedStep();
       fLinearNavigator->LocateGlobalPointAndUpdateTouchableHandle(
           track.GetPosition(), track.GetMomentumDirection(),
           newTouchable, true);

       if(newTouchable->GetVolume() == nullptr)
       {
          return 0;
       }

       State(fCurrentTouchableHandle) = newTouchable;

       //=======================================================================
       // TODO: speed up navigation update
//       geometryStepLength = fLinearNavigator->ComputeStep(track.GetPosition(),
//                                     track.GetMomentumDirection(),
//                                     currentMinimumStep,
//                                     currentSafety);
       //=======================================================================


       //=======================================
       // Longer but safer ...
       geometryStepLength =
             G4ITTransportation::AlongStepGetPhysicalInteractionLength(
                 track, previousStepSize, currentMinimumStep, currentSafety,
                 selection);

      }
  }

  //============================================================================
  // DEBUG
 // G4cout << "geometryStepLength: " << G4BestUnit(geometryStepLength, "Length")
 //        << " | trackID: " << track.GetTrackID()
 //        << G4endl;
  //============================================================================
//Hoang exp
  if(fpUserBrownianAction != nullptr)
  {
    // Let the user Brownian action class decide what to do
    G4double distanceToBoundary = fpUserBrownianAction->GetDistanceToBoundary(track);
    geometryStepLength = distanceToBoundary;
  }

//Hoang exp


  G4double diffusionCoefficient = 0;

  // Ice-specific: Use position-dependent Arrhenius diffusion for OH and Hydrogen
  G4Molecule* molecule = GetMolecule(track);
  
  if (molecule && molecule->GetDefinition() == G4OH::Definition()) {
    // For OH molecules, use Arrhenius temperature-dependent diffusion
    diffusionCoefficient = GetOHDiffusionCoefficient(track);
    
    // Debug: Print OH diffusion coefficient usage in AlongStepGetPhysicalInteractionLength
    // G4ThreeVector position = track.GetPosition();
    // G4double temperature = GetTemperatureAtPosition(position);
    // G4cout << "DEBUG-ARRHENIUS-ALONGSTEP: OH at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
    //        << " K, D=" << diffusionCoefficient/(m2/s) << " m2/s" << G4endl;
  } else if (molecule && molecule->GetDefinition() == G4Hydrogen::Definition()) {
    // For Hydrogen molecules, use Arrhenius temperature-dependent diffusion
    diffusionCoefficient = GetHydrogenDiffusionCoefficient(track);
    
    // Debug: Print H diffusion coefficient usage in AlongStepGetPhysicalInteractionLength
    // G4ThreeVector position = track.GetPosition();
    // G4double temperature = GetTemperatureAtPosition(position);
    // G4cout << "DEBUG: H AlongStep at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
    //        << " K, D=" << diffusionCoefficient/(m2/s) << " m2/s" << G4endl;
  } else {
    // For non-OH/non-H molecules, use standard GEANT4 diffusion
    diffusionCoefficient = molecule->GetDiffusionCoefficient();
  }
  
  // To avoid divide by zero of diffusionCoefficient  
  if(diffusionCoefficient <= 0)
  {
    State(fGeometryLimitedStep) = false;
    State(theInteractionTimeLeft) = DBL_MAX;
    State(fTransportEndPosition) = track.GetPosition();
    return 0;
  }
  

  State(fComputeLastPosition) = false;
  State(fTimeStepReachedLimit) = false;

  if (State(fGeometryLimitedStep))
  {
    // 95 % of the space step distribution is lower than
    // d_95 = 2 * sqrt(2*D*t)
    // where t is the corresponding time step
    // so by inversion :
    if (fUseMaximumTimeBeforeReachingBoundary)
    {
      if(fSpeedMeUp)
      {
      State(theInteractionTimeLeft) = (geometryStepLength * geometryStepLength)
          / (diffusionCoefficient); // d_50 - use straight line
      }
      else
      {
         State(theInteractionTimeLeft) = (currentSafety * currentSafety)
                  / (diffusionCoefficient); // d_50 - use safety

         //=====================================================================
         // State(theInteractionTimeLeft) = (currentSafety * currentSafety)
         //          / (8 * diffusionCoefficient); // d_95
         //=====================================================================
      }
      State(fComputeLastPosition) = true;
    }
    else
    // Will use a random time - this is precise but long to compute in certain
    // circumstances (many particles - small volumes)
    {
      State(fRandomNumber) = G4UniformRand();
      State(theInteractionTimeLeft) = 1 / (4 * diffusionCoefficient)
          * pow(geometryStepLength / InvErfc(State(fRandomNumber)),2);

      State(fTransportEndPosition) = geometryStepLength*
          track.GetMomentumDirection() + track.GetPosition();
    }

    if (fUseSchedulerMinTimeSteps)
    {
      G4double minTimeStepAllowed = G4VScheduler::Instance()->GetLimitingTimeStep();
      //========================================================================
      // TODO
//      double currentMinTimeStep = G4VScheduler::Instance()->GetTimeStep();
      //========================================================================

      if (State(theInteractionTimeLeft) < minTimeStepAllowed)
      {
        State(theInteractionTimeLeft) = minTimeStepAllowed;
        State(fTimeStepReachedLimit) = true;
        State(fComputeLastPosition) = true;
      }
    }
    else if(State(theInteractionTimeLeft) < fInternalMinTimeStep)
      // TODO: find a better way when fForceLimitOnMinTimeSteps is not used
    {
      State(fTimeStepReachedLimit) = true;
      State(theInteractionTimeLeft) = fInternalMinTimeStep;
      if (fUseMaximumTimeBeforeReachingBoundary)
      {
        State(fComputeLastPosition) = true;
      }
    }

    State(fCandidateEndGlobalTime) =
        track.GetGlobalTime() + State(theInteractionTimeLeft);

    State(fEndGlobalTimeComputed) = true; // MK: ADDED ON 20/11/2014

    State(fPathLengthWasCorrected) = false;
  }
  else
  {
    // Transform geometrical step
    geometryStepLength = 2
        * sqrt(diffusionCoefficient * State(theInteractionTimeLeft))
        * InvErf(G4UniformRand());
    State(fPathLengthWasCorrected) = true;
    //State(fEndPointDistance) = geometryStepLength;
    State(fTransportEndPosition) = geometryStepLength*
              track.GetMomentumDirection() + track.GetPosition();
  }

#ifdef G4VERBOSE
  //    DEBUG
  if (fVerboseLevel > 1)
  {
  G4cout << GREEN_ON_BLUE
         << "G4DNABrownianTransportation_ice::AlongStepGetPhysicalInteractionLength = "
         << G4BestUnit(geometryStepLength, "Length")
         << " | trackID = "
         << track.GetTrackID()
         << RESET_COLOR
         << G4endl;
  }
#endif

// assert(geometryStepLength <  State(fEndPointDistance)
//  || (geometryStepLength >= State(fEndPointDistance) && State(fGeometryLimitedStep)));

  return geometryStepLength;
}

//////////////////////////////////////////////////////////////////////////
//
//   Initialize ParticleChange  (by setting all its members equal
//                               to corresponding members in G4Track)
G4VParticleChange*
G4DNABrownianTransportation_ice::AlongStepDoIt(const G4Track& track,
                                           const G4Step& step)
{
#ifdef DEBUG_MEM
  MemStat mem_first, mem_second, mem_diff;
#endif

#ifdef DEBUG_MEM
  mem_first = MemoryUsage();
#endif

  if (GetIT(track)->GetTrackingInfo()->IsLeadingStep()
      && State(fComputeLastPosition)
      && State(fGeometryLimitedStep)//Hoang added 26/8/2019
      )
  {
    //==========================================================================
    // DEBUG
    //
//     assert(fabs(State(theInteractionTimeLeft)-
//                G4VScheduler::Instance()->GetTimeStep()) < DBL_EPSILON);
    //==========================================================================

    G4double spaceStep = DBL_MAX;
    
    // Ice-specific: Use position-dependent Arrhenius diffusion for OH and Hydrogen
    G4Molecule* molecule = GetMolecule(track);
    G4double diffusionCoefficient;
    
    if (molecule && molecule->GetDefinition() == G4OH::Definition()) {
      // For OH molecules, use Arrhenius temperature-dependent diffusion
      diffusionCoefficient = GetOHDiffusionCoefficient(track);
      
      // Debug: Print OH diffusion coefficient usage in AlongStepDoIt
      // G4ThreeVector position = track.GetPosition();
      // G4double temperature = GetTemperatureAtPosition(position);
      // G4cout << "DEBUG: OH AlongStepDoIt at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
      //        << " K, D=" << diffusionCoefficient/(m2/s) << " m2/s" << G4endl;
    } else if (molecule && molecule->GetDefinition() == G4Hydrogen::Definition()) {
      // For Hydrogen molecules, use Arrhenius temperature-dependent diffusion
      diffusionCoefficient = GetHydrogenDiffusionCoefficient(track);
      
      // Debug: Print H diffusion coefficient usage in AlongStepDoIt
      // G4ThreeVector position = track.GetPosition();
      // G4double temperature = GetTemperatureAtPosition(position);
      // G4cout << "DEBUG: H AlongStepDoIt at z=" << position.z()/mm << " mm, T=" << temperature/kelvin 
      //        << " K, D=" << diffusionCoefficient/(m2/s) << " m2/s" << G4endl;
    } else {
      // For non-OH/non-H molecules, use standard GEANT4 diffusion
      diffusionCoefficient = molecule->GetDiffusionCoefficient();
    }

    G4double sqrt_2Dt= sqrt(2 * diffusionCoefficient * State(theInteractionTimeLeft));
    G4double x = G4RandGauss::shoot(0, sqrt_2Dt);
    G4double y = G4RandGauss::shoot(0, sqrt_2Dt);
    G4double z = G4RandGauss::shoot(0, sqrt_2Dt);

    if(State(theInteractionTimeLeft) <= fInternalMinTimeStep)
    {
      spaceStep = State(fEndPointDistance);
      State(fGeometryLimitedStep) = true;
    }
    else
    {
      spaceStep = sqrt(x * x + y * y + z * z);

      if(spaceStep >= State(fEndPointDistance))
      {
        State(fGeometryLimitedStep) = true;
       if (
           //fSpeedMeUp == false&&
           !fUseSchedulerMinTimeSteps
           && spaceStep >= State(fEndPointDistance))
       {
         spaceStep = State(fEndPointDistance);
       }
      }
      else
      {
        State(fGeometryLimitedStep) = false;
      }
    }

//    assert( (spaceStep <  State(fEndPointDistance) && State(fGeometryLimitedStep) == false)
//|| (spaceStep >= State(fEndPointDistance) && State(fGeometryLimitedStep)));

    // Calculate final position
    //
    State(fTransportEndPosition) = track.GetPosition()
               + spaceStep * track.GetMomentumDirection();

    //hoang exp
    if(fpUserBrownianAction != nullptr)
    {
      // Let the user Brownian action class decide what to do
      G4ThreeVector nextPosition{track.GetPosition().getX() + x,
                                 track.GetPosition().getY() + y,
                                 track.GetPosition().getZ() + z};
      fpUserBrownianAction->Transport(nextPosition);
      State(fTransportEndPosition) = nextPosition;
    }
    //hoang exp
  }

  if(fVerboseLevel != 0)
  {
    G4cout << GREEN_ON_BLUE
            << "G4DNABrownianTransportation_ice::AlongStepDoIt: "
                "GeometryLimitedStep = "
            << State(fGeometryLimitedStep)
            << RESET_COLOR
            << G4endl;
  }

//  static G4ThreadLocal G4int noCalls = 0;
//  noCalls++;

//  fParticleChange.Initialize(track);

  G4ITTransportation::AlongStepDoIt(track, step);

#ifdef DEBUG_MEM
  MemStat mem_intermediaire = MemoryUsage();
  mem_diff = mem_intermediaire-mem_first;
  G4cout << "\t\t\t >> || MEM || After calling G4ITTransportation::"
  "AlongStepDoIt for "<< track.GetTrackID() << ", diff is : "
  << mem_diff << G4endl;
#endif

  if(track.GetStepLength() != 0 // && State(fGeometryLimitedStep)
      //========================================================================
      // TODO: avoid changing direction after too small time steps
//    && (G4VScheduler::Instance()->GetTimeStep() > fInternalMinTimeStep
//        || fSpeedMeUp == false)
      //========================================================================
        )
  {
    Diffusion(track);
  }
  //else
  //{
  //  fParticleChange.ProposeMomentumDirection(State(fTransportEndMomentumDir));
  //}
/*
  if (State(fParticleIsLooping))
  {
    if ((State(fNoLooperTrials)>= fThresholdTrials))
    {
      fParticleChange.ProposeTrackStatus(fStopAndKill);
      State(fNoLooperTrials) = 0;
#ifdef G4VERBOSE
      if ((fVerboseLevel > 1))
      {
        G4cout
            << " G4DNABrownianTransportation is killing track that is looping or stuck "
            << G4endl;
        G4cout << "   Number of trials = " << State(fNoLooperTrials)
        << "   No of calls to AlongStepDoIt = " << noCalls
        << G4endl;
      }
#endif
    }
    else
    {
      State(fNoLooperTrials)++;
    }
  }
  else
  {
    State(fNoLooperTrials)=0;
  }
*/
#ifdef DEBUG_MEM
  mem_intermediaire = MemoryUsage();
  mem_diff = mem_intermediaire-mem_first;
  G4cout << "\t\t\t >> || MEM || After calling G4DNABrownianTransportation_ice::"
  "Diffusion for "<< track.GetTrackID() << ", diff is : "
  << mem_diff << G4endl;
#endif

  return &fParticleChange;
}

